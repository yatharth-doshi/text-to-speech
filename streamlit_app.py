import json
import boto3
import streamlit as st
import textwrap

from api_request_schema import api_request_list

# Initialize Polly client
polly_client = boto3.client(service_name='polly', region_name='us-east-1')

# Get available voices and map them to their language codes
def get_language_voice_mapping():
    voices = polly_client.describe_voices()['Voices']
    language_voice_mapping = {}
    for voice in voices:
        lang_code = voice['LanguageCode']
        voice_id = voice['Id']
        if lang_code not in language_voice_mapping:
            language_voice_mapping[lang_code] = []
        language_voice_mapping[lang_code].append(voice_id)
    return language_voice_mapping

language_voice_mapping = get_language_voice_mapping()

# Load configuration
model_id = 'amazon.titan-text-lite-v1'
aws_region = 'us-east-1'

# Initialize Bedrock and Translate clients
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=aws_region)
translate_client = boto3.client(service_name='translate', region_name=aws_region)

# Language options supported by both AWS Polly and AWS Translate
language_options = {
    "auto": "Auto-detect",
    "ar-AE": "Arabic (UAE)",
    "en-US": "English (US)",
    "en-IN": "English (India)",
    "es-MX": "Spanish (Mexico)",
    "en-ZA": "English (South Africa)",
    "tr-TR": "Turkish (Turkey)",
    "ru-RU": "Russian (Russia)",
    "ro-RO": "Romanian (Romania)",
    "pt-PT": "Portuguese (Portugal)",
    "pl-PL": "Polish (Poland)",
    "nl-NL": "Dutch (Netherlands)",
    "it-IT": "Italian (Italy)",
    "is-IS": "Icelandic (Iceland)",
    "fr-FR": "French (France)",
    "fi-FI": "Finnish (Finland)",
    "es-ES": "Spanish (Spain)",
    "de-DE": "German (Germany)",
    "yue-CN": "Cantonese (China)",
    "ko-KR": "Korean (South Korea)",
    "en-NZ": "English (New Zealand)",
    "en-GB-WLS": "English (Wales)",
    "hi-IN": "Hindi (India)",
    "arb": "Arabic (Modern Standard)",
    "cy-GB": "Welsh (Wales)",
    "cmn-CN": "Chinese (Mandarin, China)",
    "da-DK": "Danish (Denmark)",
    "en-AU": "English (Australia)",
    "pt-BR": "Portuguese (Brazil)",
    "nb-NO": "Norwegian (Norway)",
    "sv-SE": "Swedish (Sweden)",
    "ja-JP": "Japanese (Japan)",
    "es-US": "Spanish (US)",
    "ca-ES": "Catalan (Spain)",
    "fr-CA": "French (Canada)",
    "en-GB": "English (UK)",
    "de-AT": "German (Austria)"
}

api_request = api_request_list[model_id]

config = {
    'log_level': 'none',  # One of: info, debug, none
    'last_speech': f"What is your name?",
    'region': aws_region,
    'polly': {
        'Engine': 'neural',
        'LanguageCode': 'en-US',
        'VoiceId': 'Joanna',
        'OutputFormat': 'mp3',
    },
    'translate': {
        'SourceLanguageCode': 'en',
        'TargetLanguageCode': 'en',
    },
    'bedrock': {
        'response_streaming': True,
        'api_request': api_request
    }
}

def translate_text(text, source_lang, target_lang):
    response = translate_client.translate_text(
        Text=text,
        SourceLanguageCode=source_lang,
        TargetLanguageCode=target_lang
    )
    return response['TranslatedText']

def translate_response_text(text, source_lang, target_lang):
    response = translate_client.translate_text(
        Text=text,
        SourceLanguageCode=source_lang,
        TargetLanguageCode=target_lang
    )
    return response['TranslatedText']

# def synthesize_speech(text, filename, voice_id):
#     response = polly_client.synthesize_speech(
#         Text=text,
#         Engine=config['polly']['Engine'],
#         LanguageCode=config['polly']['LanguageCode'],
#         VoiceId=voice_id,
#         OutputFormat=config['polly']['OutputFormat']
#     )
    
#     with open(filename, 'wb') as file:
#         file.write(response['AudioStream'].read())

def synthesize_speech(text, filename, voice_id):
    # Polly can only handle up to 3000 characters per request
    max_text_length = 3000
    
    # Split the text into chunks that are each at most 3000 characters long
    text_chunks = textwrap.wrap(text, max_text_length, break_long_words=False)
    
    with open(filename, 'wb') as file:
        for chunk in text_chunks:
            response = polly_client.synthesize_speech(
                Text=chunk,
                Engine=config['polly']['Engine'],
                LanguageCode=config['polly']['LanguageCode'],
                VoiceId=voice_id,
                OutputFormat=config['polly']['OutputFormat']
            )
            file.write(response['AudioStream'].read())

class BedrockModelsWrapper:

    @staticmethod
    def define_body(text):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        body = config['bedrock']['api_request']['body']

        if model_provider == 'amazon':
            body['inputText'] = text
        else:
            raise Exception('Unknown model provider.')

        return body

    @staticmethod
    def get_stream_chunk(event):
        return event.get('chunk')

    @staticmethod
    def get_stream_text(chunk):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]

        chunk_obj = ''
        text = ''
        if model_provider == 'amazon':
            chunk_obj = json.loads(chunk.get('bytes').decode())
            text = chunk_obj['outputText']
        else:
            raise NotImplementedError('Unknown model provider.')

        return text

def invoke_bedrock(text):
    body = BedrockModelsWrapper.define_body(text)
    response = bedrock_runtime.invoke_model_with_response_stream(
        body=json.dumps(body),
        modelId=model_id,
        accept='application/json',
        contentType='application/json'
    )
    bedrock_stream = response['body']
    return to_audio_generator(bedrock_stream)

def to_audio_generator(bedrock_stream):
    prefix = ''

    if bedrock_stream:
        for event in bedrock_stream:
            chunk = BedrockModelsWrapper.get_stream_chunk(event)
            if chunk:
                text = BedrockModelsWrapper.get_stream_text(chunk)

                if '.' in text:
                    a = text.split('.')[:-1]
                    to_polly = ''.join([prefix, '.'.join(a), '. '])
                    prefix = text.split('.')[-1]
                    yield to_polly
                else:
                    prefix = ''.join([prefix, text])

        if prefix != '':
            yield f'{prefix}.'

def main():

    st.set_page_config(page_title="Text to Speech", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

    st.title("Text-to-Speech with Bedrock and Polly")

    # Text input
    transcript_text = st.text_area("Enter the transcript text:")

    # Separate input and output language options
    input_language_options = language_options.copy()
    output_language_options = {k: v for k, v in language_options.items() if k != "auto"}

    # Language selection
    input_language = st.selectbox("Select Input Language", list(input_language_options.keys()), format_func=lambda x: input_language_options[x])
    output_language = st.selectbox("Select Output Language", list(output_language_options.keys()), format_func=lambda x: output_language_options[x])

    # Voice selection based on input language
    if output_language in language_voice_mapping:
        available_voices = language_voice_mapping[output_language]
    else:
        available_voices = []

    selected_voice = st.selectbox("Select Voice", available_voices)

    # Update configuration based on user selection
    if input_language == 'auto':
        config['polly']['LanguageCode'] = 'en-US'  # Default to English (US) if auto is selected
    else:
        config['polly']['LanguageCode'] = input_language

    config['polly']['VoiceId'] = selected_voice
    config['translate']['SourceLanguageCode'] = input_language.split('-')[0]
    config['translate']['TargetLanguageCode'] = output_language.split('-')[0]

    if st.button("Process"):
        # try:
            if transcript_text:
                if input_language != 'auto':
                    translated_transcript = translate_text(transcript_text, source_lang=input_language.split('-')[0], target_lang='en')
                else:
                    translated_transcript = transcript_text

                st.write(f"Translated Transcript: {translated_transcript}")

                # Synthesize and play input transcript
                synthesize_speech(translated_transcript, 'input.mp3', selected_voice)
                st.audio('input.mp3', format="audio/mpeg", loop=False)

                # Get response from Bedrock
                response_gen = invoke_bedrock(translated_transcript)

                # Concatenate all chunks into a single response
                full_response = ''.join([chunk for chunk in response_gen])

                # Translate response to the desired language
                translated_response = translate_response_text(full_response, source_lang='en', target_lang=output_language.split('-')[0])
                st.write(f"Translated Response: {translated_response}")

                # Synthesize and play response
                synthesize_speech(translated_response, 'response.mp3', selected_voice)
                st.audio('response.mp3', format="audio/mpeg", loop=False)
            else:
                st.error("Please enter the transcript text.")
        # except Exception as e:
        #     st.write("This voice does not support the selected engine: neural")

if __name__ == '__main__':
    main()
