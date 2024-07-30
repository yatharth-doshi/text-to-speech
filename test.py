import boto3

# Create a Polly client
polly = boto3.client("polly", region_name="us-east-1")

# List all available voices
def list_all_voices():
    try:
        response = polly.describe_voices()
        return response['Voices']
    except Exception as e:
        print(f"Error listing voices: {e}")
        return []

# List voices for a specific language
def list_voices_by_language(language_code):
    try:
        response = polly.describe_voices(LanguageCode=language_code)
        return response['Voices']
    except Exception as e:
        print(f"Error listing voices for language {language_code}: {e}")
        return []

# Example usage
# all_voices = list_all_voices()
# for voice in all_voices:
#     print(f"Voice Name: {voice['Name']}, Language: {voice['LanguageName']}")

english_voices = list_voices_by_language('da-DK')
for voice in english_voices:
    print(f"Voice Name: {voice['Name']}, Language: {voice['LanguageName']}")
