# 0- libraries
import transformers
import gradio as gr

from youtube_transcript_api import YouTubeTranscriptApi
from huggingface_hub import InferenceClient
from pytube import YouTube
import pytube
import torch

# 1 - abstractive_summary
# 1.1 - initialize
import os
save_dir = os.path.join(os.getcwd(), "docs")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
transcription_model_id = "openai/whisper-large"
llm_model_id = "tiiuae/falcon-7b-instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# 1.2 - transcription
def get_yt_transcript(url):
    text = ""
    vid_id = pytube.extract.video_id(url)
    temp = YouTubeTranscriptApi.get_transcript(vid_id)
    for t in temp:
        text += t["text"] + " "
    return text

# 1.2.1 - locally_transcribe
def transcribe_yt_vid(url):
    # download YouTube video's audio
    yt = YouTube(str(url))
    audio = yt.streams.filter(only_audio=True).first()
    out_file = audio.download(filename="audio.mp3", output_path=save_dir)

    # defining an automatic-speech-recognition pipeline
    asr = transformers.pipeline(
        "automatic-speech-recognition",
        model=transcription_model_id,
        device_map="auto",
    )

    # setting model config parameters
    asr.model.config.forced_decoder_ids = asr.tokenizer.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )

    # invoking the Whisper model
    temp = asr(out_file, chunk_length_s=20)
    text = temp["text"]

    # we can do this at the end to release GPU memory
    del asr
    torch.cuda.empty_cache()

    return text

# 1.2.1 - api_transcribe
def transcribe_yt_vid_api(url, api_token):
    # download YouTube video's audio
    yt = YouTube(str(url))
    audio = yt.streams.filter(only_audio=True).first()
    out_file = audio.download(filename="audio.wav", output_path=save_dir)

    # Initialize client for the Whisper model
    client = InferenceClient(model=transcription_model_id, token=api_token)

    import librosa
    import soundfile as sf

    text = ""
    t = 25  # audio chunk length in seconds
    x, sr = librosa.load(out_file, sr=None)
    # This gives x as audio file in numpy array and sr as original sampling rate
    # The audio needs to be split in 20 second chunks since the API call truncates the response
    for _, i in enumerate(range(0, (len(x) // (t * sr)) + 1)):
        y = x[t * sr * i : t * sr * (i + 1)]
        split_path = os.path.join(save_dir, "audio_split.wav")
        sf.write(split_path, y, sr)
        text += client.automatic_speech_recognition(split_path)

    return text


# 1.2.3 - transcribe locally or api
def transcribe_youtube_video(url, force_transcribe=False, use_api=False, api_token=None):

    yt = YouTube(str(url))
    text = ""
    # get the transcript from YouTube if available
    try:
        text = get_yt_transcript(url)
    except:
        pass

    # transcribes the video if YouTube did not provide a transcription
    # or if you want to force_transcribe anyway
    if text == "" or force_transcribe:
        if use_api:
            text = transcribe_yt_vid_api(url, api_token=api_token)
            transcript_source = "The transcript was generated using {} via the Hugging Face Hub API.".format(
                transcription_model_id
            )
        else:
            text = transcribe_yt_vid(url)
            transcript_source = (
                "The transcript was generated using {} hosted locally.".format(
                    transcription_model_id
                )
            )
    else:
        transcript_source = "The transcript was downloaded from YouTube."

    return yt.title, text, transcript_source


# 1.3 - turn to paragraph or points
def turn_to_paragraph(text):
    # REMOVE HTML TAGS
    from bs4 import BeautifulSoup

    # Parse the HTML text
    soup = BeautifulSoup(text, "html.parser")
    # Get the text without HTML tags
    text = soup.get_text()  # print(text_without_tags)

    # Remove leading and trailing whitespaces
    text = text.strip()
    # Check if the string ends with "User" and remove it
    if text.endswith("User"):
        text = text[: -len("User")]
    # Replace dashes and extra whitespaces with spaces
    text = (
        text.replace(" -", "")
        .replace("  ", "")
        .replace("\n", " ")
        .replace("- ", "")
        .replace("`", "")
    )
    # text = text.replace("  ", "\n\n") # to keep second paragraph if it exists # sometime it's good to turn this on. but let's keep it off
    text = text.replace("  ", " ")  # off this if ^ is on

    return text


# 1.3.1
def turn_to_points(text):  # input must be from `turn_to_paragraph()`
    # text = text.replace(". ", ".\n-") # to keep second paragraph if it exists
    text_with_dashes = ".\n".join("- " + line.strip() for line in text.split(". "))
    text_with_dashes = text_with_dashes.replace("\n\n", "\n\n- ")  # for first sentence of new paragraph
    return text_with_dashes

# 1.3.2 - combined funtions above for paragraph_or_points
def paragraph_or_points(text, pa_or_po):
    if pa_or_po == "Points":
        return turn_to_points(turn_to_paragraph(text))
    else:  # default is Paragraph
        return turn_to_paragraph(text)

# 1.4 - summarization
def summarize_text(title, text, temperature, words, use_api=False, api_token=None, do_sample=False, length="Short", pa_or_po="Paragraph",):

    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    import torch
    import transformers
    from transformers import BitsAndBytesConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from langchain import HuggingFacePipeline
    import torch

    model_kwargs1 = {
        "temperature": temperature,
        "do_sample": do_sample,
        "min_new_tokens": 200 - 25,
        "max_new_tokens": 200 + 25,
        "repetition_penalty": 20.0,
    }
    model_kwargs2 = {
        "temperature": temperature,
        "do_sample": do_sample,
        "min_new_tokens": words,
        "max_new_tokens": words + 100,
        "repetition_penalty": 20.0,
    }
    if not do_sample:
        del model_kwargs1["temperature"]
        del model_kwargs2["temperature"]

    if use_api:

        from langchain import HuggingFaceHub

        # os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token
        llm = HuggingFaceHub(
            repo_id=llm_model_id,
            model_kwargs=model_kwargs1,
            huggingfacehub_api_token=api_token,
        )
        llm2 = HuggingFaceHub(
            repo_id=llm_model_id,
            model_kwargs=model_kwargs2,
            huggingfacehub_api_token=api_token,
        )
        summary_source = (
            "The summary was generated using {} via Hugging Face API.".format(
                llm_model_id
            )
        )

    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            # quantization_config=quantization_config
        )
        model.to_bettertransformer()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            pad_token_id=tokenizer.eos_token_id,
            **model_kwargs1,
        )
        pipeline2 = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            pad_token_id=tokenizer.eos_token_id,
            **model_kwargs2,
        )
        llm = HuggingFacePipeline(pipeline=pipeline)
        llm2 = HuggingFacePipeline(pipeline=pipeline2)

        summary_source = "The summary was generated using {} hosted locally.".format(
            llm_model_id
        )

    # Map
    map_template = """
    Summarize the following video in a clear way:\n
    ----------------------- \n
    TITLE: `{title}`\n
    TEXT:\n
    `{docs}`\n
    ----------------------- \n
    SUMMARY:\n
    """
    map_prompt = PromptTemplate(
        template=map_template, input_variables=["title", "docs"]
    )
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce - Collapse
    collapse_template = """
    TITLE: `{title}`\n
    TEXT:\n
    `{doc_summaries}`\n
    ----------------------- \n
    Turn the text of a video above into a long essay:\n
    """

    collapse_prompt = PromptTemplate(
        template=collapse_template, input_variables=["title", "doc_summaries"]
    )
    collapse_chain = LLMChain(llm=llm, prompt=collapse_prompt)  # LLM 1 <-- LLM

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    collapse_documents_chain = StuffDocumentsChain(
        llm_chain=collapse_chain, document_variable_name="doc_summaries"
    )

    # Final Reduce - Combine
    combine_template_short = """\n
    TITLE: `{title}`\n
    TEXT:\n
    `{doc_summaries}`\n
    ----------------------- \n
    Turn the text of a video above into a 3-sentence summary:\n
    """
    combine_template_medium = """\n
    TITLE: `{title}`\n
    TEXT:\n
    `{doc_summaries}`\n
    ----------------------- \n
    Turn the text of a video above into a long summary:\n
    """
    combine_template_long = """\n
    TITLE: `{title}`\n
    TEXT:\n
    `{doc_summaries}`\n
    ----------------------- \n
    Turn the text of a video above into a long essay:\n
    """
    # Turn the text of a video above into a 3-sentence summary:\n
    # Turn the text of a video above into a long summary:\n
    # Turn the text of a video above into a long essay:\n
    if length == "Medium":
        combine_prompt = PromptTemplate(
            template=combine_template_medium,
            input_variables=["title", "doc_summaries", "words"],
        )
    elif length == "Long":
        combine_prompt = PromptTemplate(
            template=combine_template_long,
            input_variables=["title", "doc_summaries", "words"],
        )
    else:  # default is short
        combine_prompt = PromptTemplate(
            template=combine_template_short,
            input_variables=["title", "doc_summaries", "words"],
        )
    combine_chain = LLMChain(llm=llm2, prompt=combine_prompt)  # LLM 2 <-- LLM2

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=combine_chain, document_variable_name="doc_summaries"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=collapse_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=800,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import TokenTextSplitter

    with open(save_dir + "/transcript.txt", "w") as f:
        f.write(text)
    loader = TextLoader(save_dir + "/transcript.txt")
    doc = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(doc)

    summary = map_reduce_chain.run(
        {"input_documents": docs, "title": title, "words": words}
    )

    try:
        del (map_reduce_chain, reduce_documents_chain,
            combine_chain, collapse_documents_chain,
            map_chain, collapse_chain,
            llm, llm2,
            pipeline, pipeline2,
            model, tokenizer)
    except:
        pass
    torch.cuda.empty_cache()

    summary = paragraph_or_points(summary, pa_or_po)

    return summary, summary_source


# 1.5 - complete function [DELETED]

# 2 - extractive/low-abstractive summary for Key Sentence Highlight
# 2.1 - chunking + hosted inference, summary [DELETED]

# 2.2 - add spaces between punctuations
import re
def add_space_before_punctuation(text):
    # Define a regular expression pattern to match punctuation
    punctuation_pattern = r"([.,!?;:])"

    # Use re.sub to add a space before each punctuation
    modified_text = re.sub(punctuation_pattern, r" \1", text)

    bracket_pattern = r'([()])'
    modified_text = re.sub(bracket_pattern, r" \1 ", modified_text)

    return modified_text


# 2.3 - highlight same words (yellow)
from difflib import ndiff
def highlight_text_with_diff(text1, text2):
    diff = list(ndiff(text1.split(), text2.split()))

    highlighted_diff = []
    for item in diff:
        if item.startswith(" "):
            highlighted_diff.append(
                '<span style="background-color: rgba(255, 255, 0, 0.25);">'
                + item
                + " </span>"
            )  # Unchanged words
        elif item.startswith("+"):
            highlighted_diff.append(item[2:] + " ")

    return "".join(highlighted_diff) # output in string HTML format

# 2.4 - combined - `highlight_key_sentences`
#   extractive/low-abstractive summarizer with facebook/bart-large-cnn
#   highlight feature
def highlight_key_sentences(original_text, api_key):

    import requests

    API_TOKEN = api_key
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    def chunk_text(text, chunk_size=1024):
        # Split the text into chunks
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks

    def summarize_long_text(long_text):
        # Split the long text into chunks
        text_chunks = chunk_text(long_text)

        # Summarize each chunk
        summaries = []
        for chunk in text_chunks:
            data = query(
                {
                    "inputs": f"{chunk}",
                    "parameters": {"do_sample": False},
                }
            )  # what if do_sample=True?
            summaries.append(data[0]["summary_text"])

        # Combine the summaries of all chunks
        full_summary = " ".join(summaries)
        return full_summary

    summarized_text = summarize_long_text(original_text)

    original_text = add_space_before_punctuation(original_text)
    summarized_text = add_space_before_punctuation(summarized_text)

    return highlight_text_with_diff(summarized_text, original_text)  # output in string HTML format


# 3 - extract_keywords
# 3.1 - initialize & load pipeline
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs,
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])


# Load pipeline
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)

# 3.2 - re-arrange keywords order
import re
def rearrange_keywords(text, keywords):  # text:str, keywords:List
    # Find the positions of each keyword in the text
    keyword_positions = {word: text.lower().index(word.lower()) for word in keywords}

    # Sort the keywords based on their positions in the text
    sorted_keywords = sorted(keywords, key=lambda x: keyword_positions[x])

    return sorted_keywords

# 3.3 - `keywords_extractor` function
def keywords_extractor_list(summary):  # List  : Flashcards
    keyphrases = extractor(summary)  # extractor() from above | text.replace("\n", " ")
    list_keyphrases = keyphrases.tolist()

    # rearrange first
    list_keyphrases = rearrange_keywords(summary, list_keyphrases)

    return list_keyphrases  # returns List

def keywords_extractor_str(summary):  # str   : Keywords Highlight & Fill in the Blank
    keyphrases = extractor(summary)  # extractor() from above | text.replace("\n", " ")
    list_keyphrases = keyphrases.tolist()

    # rearrange first
    list_keyphrases = rearrange_keywords(summary, list_keyphrases)

    # join List elements to one string
    all_keyphrases = " ".join(list_keyphrases)

    return all_keyphrases  # returns one string

# 3.4 - keywords highlight
# 3.4.1 - highlight same words (green)
def highlight_green(text1, text2):  # keywords(str), text
    diff = list(ndiff(text1.split(), text2.split()))

    highlighted_diff = []
    for item in diff:
        if item.startswith(" "):
            highlighted_diff.append(
                '<span style="background-color: rgba(0, 255, 0, 0.25);">'
                + item
                + " </span>"
            )  # Unchanged words
        elif item.startswith("+"):
            highlighted_diff.append(item[2:] + " ")

    return "".join(highlighted_diff) # output in string HTML format


# 3.4.2 - combined - keywords highlight
def keywords_highlight(text):
    keywords = keywords_extractor_str(text) # keywords; one string
    text = add_space_before_punctuation(text)
    return highlight_green(keywords, text) # output in string HTML format


# 3.5 - flashcards
# 3.5.1 - pair_keywords_sentences
def pair_keywords_sentences(text, search_words):  # text:str, search_words:List

    result_html = "<span style='text-align: center;'>"

    # Split the text into sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    # Create a dictionary to store sentences for each keyword
    keyword_sentences = {word: [] for word in search_words}

    # Iterate through sentences and search for keywords
    for sentence in sentences:
        for word in search_words:
            if re.search(
                r"\b{}\b".format(re.escape(word)), sentence, flags=re.IGNORECASE
            ):
                keyword_sentences[word].append(sentence)

    # Print the results
    for word, sentences in keyword_sentences.items():
        result_html += "<h2>" + word + "</h2> \n"

        for sentence in sentences:
            result_html += "<p>" + sentence + "</p> \n"

        result_html += "\n"

    result_html += "</span>"

    return result_html

# 3.5.2 combined - flashcards
def flashcards(text):
    keywords = keywords_extractor_list(text) # keywords; a List
    text = add_space_before_punctuation(text)
    return pair_keywords_sentences(text, keywords) # output in string HTML format


# 3.6 - fill in the blank
# 3.6.1 - underline same words
def underline_keywords(text1, text2):  # keywords(str), text
    diff = list(ndiff(text1.split(), text2.split()))

    highlighted_diff = []
    for item in diff:
        if item.startswith(" "):
            highlighted_diff.append(
                "_______"
            )  # Unchanged words. make length independent of word length?
        elif item.startswith("+"):
            highlighted_diff.append(item[2:] + " ")

    return "".join(highlighted_diff) # output in string HTML format


# 3.6.2 - combined - underline
def fill_in_blanks(text):
    keywords = keywords_extractor_str(text)  # keywords; one string
    text = add_space_before_punctuation(text)
    return underline_keywords(keywords, text)  # output in string HTML format


# 4 - misc
emptyTabHTML = "<br>\n<p style='color: gray; text-align: center'>Please generate a summary first.</p>\n<br>\n<br>\n<br>\n<br>\n<br>\n<br>\n<br>\n<br>\n<br>\n<br>\n<br>\n<br>\n<br>\n"


def empty_tab():
    return emptyTabHTML


# 5 - the app
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("<br>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("# ✍️ Summarizer for Learning")
        with gr.Column():
            gr.HTML("<div style='color: red; text-align: right'>Please use your <a href='#HFAPI' style='color: red'>Hugging Face Access Token.</a></div>")

    with gr.Row():
        with gr.Column():
            with gr.Tab("YouTube"):
                yt_link = gr.Textbox(show_label=False, placeholder="Insert YouTube link here ... (video needs to have caption)")
                yt_transcript = gr.Textbox(show_label=False, placeholder="Transcript will be shown here ...", lines=12)
            with gr.Tab("Article"):
                gr.Textbox(show_label=False, placeholder="WORK IN PROGRESS", interactive=False)
                gr.Textbox(show_label=False, placeholder="", lines=12, interactive=False)
            with gr.Tab("Text"):
                gr.Dropdown(["WORK IN PROGRESS", "Example 2"], show_label=False, value="WORK IN PROGRESS", interactive=False)
                gr.Textbox(show_label=False, placeholder="", lines=12, interactive=False)
            with gr.Row():
                clrButton = gr.ClearButton([yt_link, yt_transcript])
                subButton = gr.Button(variant="primary", value="Summarize")

            with gr.Accordion("Settings", open=False):
                length = gr.Radio(["Short", "Medium", "Long"], label="Length", value="Short", interactive=True)
                pa_or_po = gr.Radio(["Paragraphs", "Points"], label="Summarize to", value="Paragraphs", interactive=True)
                gr.Checkbox(label="Add headings", interactive=False)
                gr.Radio(["One section", "Few sections"], label="Summarize into", interactive=False)  # info="Only for 'Medium' or 'Long'"
                with gr.Row():
                    clrButtonSt1 = gr.ClearButton([length, pa_or_po], interactive=True)
                    subButtonSt1 = gr.Button(value="Set Current as Default", interactive=False)
                    subButtonSt1 = gr.Button(value="Show Default", interactive=False)

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Group(visible=False):
                    gr.HTML("<p style='text-align: center;'>&nbsp; YouTube transcription</p>")
                    force_transcribe_with_app = gr.Checkbox(
                        label="Always transcribe with app",
                        info="The app first checks if caption on YouTube is available. If ticked, the app will transcribe the video for you but slower.",
                    )
                with gr.Group():
                    gr.HTML("<p style='text-align: center;'>&nbsp; Summarization</p>")
                    gr.Radio(["High Abstractive", "Low Abstractive", "Extractive"], label="Type of summarization", value="High Abstractive", interactive=False)
                    gr.Dropdown(
                        [
                            "tiiuae/falcon-7b-instruct",
                            "GPT2 (work in progress)",
                            "OpenChat 3.5 (work in progress)",
                        ],
                        label="Model",
                        value="tiiuae/falcon-7b-instruct",
                        interactive=False,
                    )
                    temperature = gr.Slider(0.10, 0.30, step=0.05, label="Temperature", value=0.15,
                        info="Temperature is limited to 0.1 ~ 0.3 window, as it is shown to produce best result.",
                        interactive=True,
                    )
                    do_sample = gr.Checkbox(label="do_sample", value=True,
                        info="If ticked, do_sample produces more creative and diverse text, otherwise the app will use greedy decoding that generate more consistent and predictable summary.",
                    )

                with gr.Group():
                    gr.HTML("<p style='text-align: center;'>&nbsp; Highlight</p>")
                    check_key_sen = gr.Checkbox(label="Highlight key sentences", info="In original text", value=True, interactive=False)
                    gr.Checkbox(label="Highlight keywords", info="In summary", value=True, interactive=False)
                    gr.Checkbox(label="Turn text to paragraphs", interactive=False)

                with gr.Group():
                    gr.HTML("<p style='text-align: center;'>&nbsp; Quiz mode</p>")
                    gr.Checkbox(label="Fill in the blanks", value=True, interactive=False)
                    gr.Checkbox(label="Flashcards", value=True, interactive=False)
                    gr.Checkbox(label="Re-write summary", interactive=False)  # info="Only for 'Short'"

                with gr.Row():
                    clrButtonSt2 = gr.ClearButton(interactive=True)
                    subButtonSt2 = gr.Button(value="Set Current as Default", interactive=False)
                    subButtonSt2 = gr.Button(value="Show Default", interactive=False)

        with gr.Column():
            with gr.Tab("Summary"):  # Output
                title = gr.Textbox(show_label=False, placeholder="Title")
                summary = gr.Textbox(lines=11, show_copy_button=True, label="", placeholder="Summarized output ...")
            with gr.Tab("Key sentences", render=True):
                key_sentences = gr.HTML(emptyTabHTML)
                showButtonKeySen = gr.Button(value="Generate")
            with gr.Tab("Keywords", render=True):
                keywords = gr.HTML(emptyTabHTML)
                showButtonKeyWor = gr.Button(value="Generate")
            with gr.Tab("Fill in the blank", render=True):
                blanks = gr.HTML(emptyTabHTML)
                showButtonFilBla = gr.Button(value="Generate")
            with gr.Tab("Flashcards", render=True):
                flashCrd = gr.HTML(emptyTabHTML)
                showButtonFlash = gr.Button(value="Generate")
            gr.Markdown("<span style='color: gray'>The app is a work in progress. Output may be odd and some features are disabled. [Learn more](https://huggingface.co/spaces/reflection777/summarizer-for-learning/blob/main/README.md).</span>")
            with gr.Group():
                gr.HTML("<p id='HFAPI' style='text-align: center;'>&nbsp; 🤗 Hugging Face Access Token [<a href='https://huggingface.co/settings/tokens'>more</a>]</p>")
                hf_access_token = gr.Textbox(
                    show_label=False,
                    placeholder="example: hf_******************************",
                    type="password",
                    info="The app does not store the token.",
                )
            with gr.Accordion("Info", open=False, visible=False):
                transcript_source = gr.Textbox(show_label=False, placeholder="transcript_source")
                summary_source = gr.Textbox(show_label=False, placeholder="summary_source")
                words = gr.Slider(minimum=100, maximum=500, value=250, label="Length of the summary")
                # words: what should be the constant value?
                use_api = gr.Checkbox(label="use_api", value=True)

            subButton.click(
                fn=transcribe_youtube_video,
                inputs=[yt_link, force_transcribe_with_app, use_api, hf_access_token],
                outputs=[title, yt_transcript, transcript_source],
                queue=True,
            ).then(
                fn=summarize_text,
                inputs=[title, yt_transcript, temperature, words, use_api, hf_access_token, do_sample, length, pa_or_po],
                outputs=[summary, summary_source],
                api_name="summarize_text",
                queue=True,
            )

            subButton.click(fn=empty_tab, outputs=[key_sentences])
            subButton.click(fn=empty_tab, outputs=[keywords])
            subButton.click(fn=empty_tab, outputs=[flashCrd])
            subButton.click(fn=empty_tab, outputs=[blanks])

            showButtonKeySen.click(
                fn=highlight_key_sentences,
                inputs=[yt_transcript, hf_access_token],
                outputs=[key_sentences],
                queue=True,
            )

            # Keywords
            showButtonKeyWor.click(fn=keywords_highlight, inputs=[summary], outputs=[keywords], queue=True)

            # Flashcards
            showButtonFlash.click(fn=flashcards, inputs=[summary], outputs=[flashCrd], queue=True)

            # Fill in the blanks
            showButtonFilBla.click(fn=fill_in_blanks, inputs=[summary], outputs=[blanks], queue=True)
    
    gr.Examples(
        examples=["https://www.youtube.com/watch?v=P6FORpg0KVo", "https://www.youtube.com/watch?v=bwEIqjU2qgk"],
        inputs=[yt_link]
    )

if __name__ == "__main__":
    demo.launch(show_api=False)
    # demo.launch(show_api=False, debug=True)
    # demo.launch(show_api=False, share=True)
