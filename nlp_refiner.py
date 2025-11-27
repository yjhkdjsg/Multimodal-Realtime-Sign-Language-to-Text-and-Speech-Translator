from openai import OpenAI
import time
import torch
import requests

client = OpenAI(
    base_url="http://192.168.18.1:1234/v1",  
    api_key="lm-studio"
)

def test_lm_studio_connection():
    """Test if LM Studio server is accessible"""
    try:
        response = requests.get("http://192.168.18.1:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            return True, f"Connected! Available models: {len(models.get('data', []))}"
        else:
            return False, f"Server responded with status {response.status_code}"
    except requests.exceptions.ConnectRefused:
        return False, "Connection refused - LM Studio not running or wrong address"
    except requests.exceptions.Timeout:
        return False, "Connection timeout - check network/firewall"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def preprocess(chars):
    vowels = set("aeiou")
    result = []
    current = ""

    for c in chars:
        if len(current) >= 2 and all(ch not in vowels for ch in current[-2:]) and c not in vowels:
            result.append(current)
            current = c
        else:
            current += c

    if current:
        result.append(current)

    return " ".join(result)

def refine_buffer(buffer):
    return refine_asl_buffer(buffer)

def refine_asl_buffer(buffer_input):
    start_time = time.time()
    if isinstance(buffer_input, str):
        if ' ' in buffer_input:
            buffer = buffer_input.split()
        else:
            buffer = list(buffer_input)
    else:
        buffer = buffer_input
    
    cleaned_buffer = [char.lower() for char in buffer if char and char.strip()]
    cleaned = ''.join(cleaned_buffer)
    
    preprocessed = preprocess(cleaned_buffer)
    
    connected, connection_msg = test_lm_studio_connection()
    if not connected:
        raise Exception(f"LM Studio connection failed: {connection_msg}")
    
    try:
        response = client.chat.completions.create(
            model="qwen2.5-7b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an ASL buffer decoder.\n"
                        "Rules:\n"
                        "- You ARE allowed to add spaces between characters to form correct English words.\n"
                        "- Do not invent new words unless absolutely necessary.\n"
                        "- Do not add meaning that isn't implied.\n"
                        "- Only remove repeated characters when they don't belong.\n"
                        "- Output only the corrected English sentence.\n"
                        "- Keep it concise and natural.\n"
                    )
                },
                {
                    "role": "user",
                    "content": f"Decode this ASL letter sequence into proper English: {preprocessed}"
                }
            ],
            temperature=0.1
        )
        
        refined_text = response.choices[0].message.content.strip()
        
    except Exception as e:
        raise Exception(f"LM Studio API error: {str(e)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processing_time = time.time() - start_time
    
    return {
        'refined_text': refined_text,
        'preprocessed': preprocessed,
        'cleaned': cleaned,
        'processing_time_seconds': round(processing_time, 3),
        'model_device': device,
        'connection_status': connection_msg
    }

if __name__ == "__main__":
    buffer = ['i','f','e','e','e','l','s','a','d','d']
    result = refine_asl_buffer(buffer)
    print("Original:", ''.join(buffer))
    print("Refined:", result['refined_text'])
    print("Processing time:", result['processing_time_seconds'], "seconds")
