# YouTube Video Speech to Report Converter

- This app converts speech from YouTube videos into summary reports.
- It is ideal for busy business persons or students who want to quickly review conference or lecture videos.

# Features

- Can extract keywords from the speech and includes a comprehension test
- Can translate the summary to the languages supported by OpenAI API including Japanese
- Can automatically compress extracted audio files from YouTube videos to save OpenAI usage fee
- Can automatically split too large audio files to fit OpenAI API limitations
- Can automatically split too long script to fit OpenAI API limitations
- Can summarize everything in a concise text file

# Usage

- For normal users:

  - Install ffmpeg and add it to your PATH.
  - Generate an OpenAI API key by referring to the URL below and write it in the `apikey.txt` file. The file should be in the same directory as `main.py` file.
    - https://openaimaster.com/how-to-use-openai-api-key/#Generating_an_API_Key
  - Use a command line interface to execute the file
    - e.g.) `python main.py -u https://www.youtube.com/watch?v=0O2Rq4HJBxw -s 1000 -k 5 -q 5 -t -l Japanese`
  - Type `python main.py -h` for help.
