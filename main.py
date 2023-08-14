import argparse
import tempfile
import os
from pathlib import Path
import glob
import json
import time
import subprocess

import openai
from yt_dlp import YoutubeDL
import tiktoken

# max token length in openai api
MAX_CONTENT_LENGTH = 4097

def download_youtube(url, savepath):
    """download YouTube audio file

    Args:
        url (str): url of YouTube video
        savepath (str): save path of the video
    """    
    option = {
            'outtmpl' : os.path.join(savepath, '%(title)s.%(ext)s'),
            'format' : 'bestaudio',
            # 'format' : '[asr<=44100][abr<=64000]',
        }
    
    # if it's a playlist, remove the playlist part
    if '&list=' in url:
        url = url.split('&list=')[0]

    with YoutubeDL(option) as ydl:
        ydl.download(url)

def get_openai_apikey():
    """get API key of openai (it only works in my computer. debugging purpose only)

    Raises:
        Exception: key wasn't found

    Returns:
        str: API key
    """    
    path = 'apikey.txt'
    if os.path.exists(path): # localhost
        with open(path, 'r') as f:
            return f.read().replace('\n', '')
    else:
        raise Exception('api key file does not exit')
    
def speech_to_text(file):
    """convert a speech audio file into a text

    Args:
        file (str): speech audio file path

    Returns:
        str: a converted text
    """    
    with open(file, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
    return transcript['text']

def call_chatgpt(messages):
    """call chatGPT API

    Args:
        messages (List[Dict[str, str]]): messages to send to chatGPT

    Returns:
        str: response from chatGPT
    """    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    txt = response["choices"][0]["message"]["content"]
    return txt

def call_chatgpt_translate(txt, lang):
    """translate the input to a specific language using chatGPT

    Args:
        txt (str): text to be translated

    Returns:
        str: translated text
    """    
    translated = call_chatgpt([
            {"role": "system", "content": f'You are a professional translator. You translate English to {lang}.'},
            {"role": "user", "content": f'Translate the following text to {lang}. \n\n# Text\n{txt}\n\n# Translation'},
        ])
    
    return translated

def get_n_tokens(text):
    # return the number of tokens used in chatgpt api
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    count = len(encoding.encode(text))
    return count

def split_by_token(text, block_size, sep):
    # split `text` by `block_size` at `sep`
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    lines = text.split(sep)
    blocks = []
    tokens = []
    token = 0
    block = ''
    for line in lines:
        t = len(encoding.encode(line))
        if token + t > block_size:
            blocks.append(block)
            tokens.append(token)
            token = 0
            block = '' 
        token += t
        block += line + sep
    blocks.append(block)
    tokens.append(token)
    return blocks, tokens

def summarize_text_helper(text, n_tokens):
    content = f'''# Command
- Please summarize the text in {n_tokens} tokens. The total number of tokens should be as close to {n_tokens} as possible. Be careful. Don't make any mistakes.

# Text
{text}
'''

    extract = call_chatgpt([
            {"role": "system", "content": 'You are a professor in a university. You give lectures to students. You are very good at teaching.'},
            {"role": "user", "content": content},
        ])
    return extract

def summarize_text(text):
    # summarize text to `MAX_CONTENT_LENGTH` 
    blocks, tokens = split_by_token(text, MAX_CONTENT_LENGTH * 0.9, '.')
    n_tokens_pars = [int(MAX_CONTENT_LENGTH * t / sum(tokens)) for t in tokens]
    texts = [summarize_text_helper(text, n_tokens) for text, n_tokens in zip(blocks, n_tokens_pars)]
    return ' '.join(texts)

def process_text(summary_script, whisper_script, is_token_over, summary_n_words, n_keywords, n_qa, lang, do_translate):
    """process script with chatGPT and return processed texts. "Process" means creating summary, extracting keywords and generating questions and answers for comprehension test.

    Args:
        script (str): a text to be processed
        summary_n_words (int): number of words for summary
        n_keywords (int): number of keywords
        n_qa (int): number of the pairs of questions and answers
        do_translate (bool): whether translation to another language is done
        lang (str): another language to translate to

    Returns:
        Tuple[str, Dict]: the original script and a json dictionary including summary
    """    

    content = f'''# Command
- Please summarize the text in # Text section in {summary_n_words} words (let this be result1).
- Please pick up the {n_keywords} most important keywords from the text in # Text section as a json list (let this be result2).
- Please create {n_qa} pairs of questions and answers to test comprehension from the text in # Text section as a json list of a json list. (let this be result3).
- Please return a json string in the following format.
    {{"summary": result1, "keywords": result2, "qas": result3}}

# Text
{summary_script if is_token_over else whisper_script}
'''

    extract = call_chatgpt([
            {"role": "system", "content": 'You are a professor in a university. You give lectures to students. You are very good at teaching.'},
            {"role": "user", "content": content},
        ])
    jsondict = json.loads(extract)

    if do_translate: # if the translation is done
        # translate each content in the json dictionary
        jsondict_new = jsondict.copy()
        jsondict_new['summary'] = call_chatgpt_translate(jsondict['summary'], lang)
        jsondict_new['keywords'] = [call_chatgpt_translate(k, lang) for k in jsondict['keywords']]
        jsondict_new['qas'] = [[call_chatgpt_translate(qa[0], lang), call_chatgpt_translate(qa[1], lang)] for qa in jsondict['qas']]
        jsondict_new['script'] = call_chatgpt_translate(summary_script if is_token_over else whisper_script, lang)
        jsondict_new['original'] = whisper_script
        return jsondict_new
    else:
        jsondict['script'] = whisper_script
        return jsondict

def create_and_save_report(jsondict, file, do_translate):
    """create and save a report

    Args:
        script (str): the original script generated with speech-to-text
        jsondict (Dict): a json dictionary including summary
        file (str): a name of a downloaded YouTube audio file
        do_translate (bool): whether the api translates the script
    """

    # the file name without its extension in a path
    fname = Path(file).stem
    # questions and answers
    qs = '\n'.join([f'Q{i + 1}: {qa[0]}' for i, qa in enumerate(jsondict['qas'])])
    as_ = '\n'.join([f'A{i + 1}: {qa[1]}' for i, qa in enumerate(jsondict['qas'])])
    # report template
    report = f'''========= {fname} =========
# Summary
{jsondict['summary']}

# Keywords
{', '.join(jsondict['keywords'])}

# Main text
{jsondict['script']}

# Quiz
{qs}

# Answer
{as_}
'''
    if do_translate:
        report += f'''
# Original script
{jsondict['original']}
'''
    
    # save a file
    with open(fname + '.txt', 'w', encoding='utf-8') as f:
        f.write(report)

def cut_audio(output_file, savepath, output_format):
    # get file info
    cmd = 'ffprobe -v quiet -print_format json -show_format -show_streams'.split() + [output_file]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    info = json.loads(result.stdout)
    duration = float(info['format']['duration'])
    size = int(info['format']['size'])
    # cut audio to fit openai limitation
    OPENAI_LIMIT = 26214400 * 0.9 # Maximum content size limit (26214400) * safety coefficient
    if size > OPENAI_LIMIT:
        cut_duration = int(OPENAI_LIMIT / size * duration)
        # cut audio
        cmd = f'ffmpeg -i "{output_file}" -f segment -segment_time {cut_duration} {savepath}/{output_format}'
        subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

def summarize_youtube_video(args):
    """do a series of processes

    Args:
        args (List): a list of parameters in window variable
    """    

    # download youtube audios
    print('downloading youtube audios...')
    for url in args.urls:
        print(f'- {url}')
        with tempfile.TemporaryDirectory() as savepath:
            assert ' ' not in savepath # for subprocess

            # audio from youtube video is saved
            download_youtube(url, savepath)
        
            # convert audio files into scripts
            pathjoin = os.path.join(savepath, '*')
            # a url can have multiple videos
            audiofiles = glob.glob(pathjoin)

            for file in audiofiles:
                print('audio file:', file)
                # input / output files
                input_file = Path(file)
                output_file = str(input_file.parent / input_file.stem) + '_down.webm'

                # compress audio
                print('compressing...')
                cmd = ['ffmpeg', '-i', input_file, '-y', output_file, '-ar', '44100', '-b:a', '64k']
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

                # cut audio to fit openai limitation
                output_format = 'output_%d.webm'
                output_format_glob = 'output_*.webm'
                print('cutting...')
                cut_audio(output_file, savepath, output_format)

                # extract text by whisper
                print('extracting texts by whisper...')
                whisper_scripts = []
                for f in glob.glob(str(Path(savepath) / output_format_glob)):
                    script = speech_to_text(f)
                    whisper_scripts.append(script)
                    time.sleep(1)
                whisper_script = ' '.join(whisper_scripts)

                # format text
                print('formatting texts...')
                # if the script is too long for openai api, shorten it.
                is_token_over = get_n_tokens(whisper_script) >= MAX_CONTENT_LENGTH
                summary_script = summarize_text(whisper_script) if is_token_over else whisper_script
                jsondict = process_text(summary_script, whisper_script, is_token_over, args.summary_n_words, args.n_keywords, args.n_qa, args.lang, args.do_translate)
                
                # save
                print('saving texts...')
                create_and_save_report(jsondict, input_file, args.do_translate)

                print('url done:', url)

if __name__ == '__main__':
    # create parser
    parser = argparse.ArgumentParser(
                prog='argparseTest.py', # name of the program
                usage='how to use the program', # how to use the program
                description='description', # show before help
                epilog='end', # shown after help
                add_help=True, # whether the program adds -h/–help 
                )
    
    # arguments
    parser.add_argument('-u', '--urls', nargs='*', required=True, type=str, help='urls for YouTube videos')
    parser.add_argument('-s', '--summary-n-words', default=1000, type=int, help='the number of words the video script is summarized in')
    parser.add_argument('-k', '--n-keywords', default=5, type=int, help='the number of keywords from the video script')
    parser.add_argument('-q', '--n-qa', default=5, type=int, help='the number of questions and answers from the video script')
    parser.add_argument('-t', '--do_translate', action='store_true', help='whether the program translates the video script.')
    parser.add_argument('-l', '--lang', type=str, help='the language the video script is translated to.')

    # analyze parameters
    args = parser.parse_args()

    # api key
    openai.api_key = get_openai_apikey()

    # summarize youtube video
    summarize_youtube_video(args)