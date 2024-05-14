from groundingdino.util.inference import load_model, load_image, predict, annotate
import os
import supervision as sv
from flask import Flask, request, jsonify 
import os
from transformers import AutoTokenizer, BitsAndBytesConfig #type: ignore
from llava.model import LlavaLlamaForCausalLM #type: ignore
from llava.utils import disable_torch_init #type: ignore
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN #type: ignore
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria #type: ignore
from llava.conversation import conv_templates, SeparatorStyle #type: ignore
import torch #type: ignore
from PIL import Image #type: ignore
import requests #type: ignore
from io import BytesIO #type: ignore
from PIL import Image #type: ignore
import base64
from urllib.parse import urlparse
import cv2

CONFIG_PATH = ("/home/easgrad/rkhatik/Raj/Grounding_Dino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = "/home/easgrad/rkhatik/Raj/Grounding_Dino/GroundingDINO/weights/groundingdino_swint_ogc.pth"




app = Flask(__name__)
DINOmodel = load_model(CONFIG_PATH, WEIGHTS_PATH)


BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25




class LLaVAChatBot:
    def __init__(self,
                 model_path: str = 'liuhaotian/llava-v1.5-7b',
                 device_map: str = 'auto',
                 load_in_8_bit: bool = True,
                 **quant_kwargs) -> None:
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.conv = None
        self.conv_img = None
        self.img_tensor = None
        self.roles = None
        self.stop_key = None
        self.load_models(model_path,
                         device_map=device_map,
                         load_in_8_bit=load_in_8_bit,
                         **quant_kwargs)

    def load_models(self, model_path: str,
                    device_map: str,
                    load_in_8_bit: bool,
                    **quant_kwargs) -> None:
        """Load the model, processor and tokenizer."""
        quant_cfg = BitsAndBytesConfig(**quant_kwargs)
        self.model = LlavaLlamaForCausalLM.from_pretrained(model_path,
                                                           low_cpu_mem_usage=True,
                                                           device_map=device_map,
                                                           load_in_8bit=load_in_8_bit,
                                                           quantization_config=quant_cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       use_fast=False)
        vision_tower = self.model.get_vision_tower()
        vision_tower.load_model()
        vision_tower.to(device='cuda')
        self.image_processor = vision_tower.image_processor
        disable_torch_init()

    def setup_image(self, img_path: str) -> None:
        """Load and process the image."""
        if img_path.startswith('http') or img_path.startswith('https'):
            response = requests.get(img_path)
            self.conv_img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            self.conv_img = Image.open(img_path).convert('RGB')
        self.img_tensor = self.image_processor.preprocess(self.conv_img,
                                                          return_tensors='pt'
                                                          )['pixel_values'].half().cuda()

    def generate_answer(self, **kwargs) -> str:
        """Generate an answer from the current conversation."""
        raw_prompt = self.conv.get_prompt()
        input_ids = tokenizer_image_token(raw_prompt,
                                          self.tokenizer,
                                          IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).cuda()
        stopping = KeywordsStoppingCriteria([self.stop_key],
                                            self.tokenizer,
                                            input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids,
                                             images=self.img_tensor,
                                             stopping_criteria=[stopping],
                                             **kwargs)
        outputs = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:]
        ).strip()
        self.conv.messages[-1][-1] = outputs

        return outputs.rsplit('</s>', 1)[0]

    def get_conv_text(self) -> str:
        """Return full conversation text."""
        return self.conv.get_prompt()

    def start_new_chat(self,
                       img_path: str,
                       prompt: str,
                       do_sample=True,
                       temperature=0.2,
                       max_new_tokens=1024,
                       use_cache=True,
                       **kwargs) -> str:
        """Start a new chat with a new image."""
        conv_mode = "v1"
        self.setup_image(img_path)
        self.conv = conv_templates[conv_mode].copy()
        self.roles = self.conv.roles
        first_input = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN +
                       DEFAULT_IM_END_TOKEN + '\n' + prompt)
        self.conv.append_message(self.roles[0], first_input)
        self.conv.append_message(self.roles[1], None)
        if self.conv.sep_style == SeparatorStyle.TWO:
            self.stop_key = self.conv.sep2
        else:
            self.stop_key = self.conv.sep
        answer = self.generate_answer(do_sample=do_sample,
                                      temperature=temperature,
                                      max_new_tokens=max_new_tokens,
                                      use_cache=use_cache,
                                      **kwargs)
        return answer

    def continue_chat(self,
                      prompt: str,
                      do_sample=True,
                      temperature=0.2,
                      max_new_tokens=1024,
                      use_cache=True,
                      **kwargs) -> str:
        """Continue the existing chat."""
        if self.conv is None:
            raise RuntimeError("No existing conversation found. Start a new"
                               "conversation using the `start_new_chat` method.")
        self.conv.append_message(self.roles[0], prompt)
        self.conv.append_message(self.roles[1], None)
        answer = self.generate_answer(do_sample=do_sample,
                                      temperature=temperature,
                                      max_new_tokens=max_new_tokens,
                                      use_cache=use_cache,
                                      **kwargs)
        return answer


app = Flask(__name__)
chatbot = LLaVAChatBot(load_in_8bit=True,
                       bnb_8bit_compute_dtype=torch.float16,
                       bnb_8bit_use_double_quant=True,
                       bnb_8bit_quant_type='nf8')

def dino_chat(image_path,prompt):

    # Load the image
    image_source, image = load_image(image_path)
    lowercase_prompt = prompt.lower()

    print("Getting coordinates")
    TEXT_PROMPT = prompt
    boxes, logits, phrases = predict(
        model=DINOmodel, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )
    print("a:",boxes,type(boxes))
    boxes_str = boxes.tolist()
    return boxes_str


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt')
    print("prompt: ",prompt)
    if 'image_base64' in data and prompt:
        try:
            # Create temp folder if it doesn't exist
            if not os.path.exists('temp'):
                os.makedirs('temp')

            # Save the image as temp/temp.jpeg
            img_data = base64.b64decode(data['image_base64'])
            with open('temp/temp.jpeg', 'wb') as img_file:
                img_file.write(img_data)

            # Use temp/temp.jpeg as img_path
            answer = chatbot.start_new_chat(img_path='temp/temp.jpeg', prompt=prompt)

            # Check if DINO chat should be executed
            lowercase_prompt = prompt.lower()
            if any(word in lowercase_prompt for word in ["locate", "track", "coordinates", "location", "position", "go to"]):
                coordinates = dino_chat(image_path='temp/temp.jpeg',prompt= prompt)
                answer = coordinates
                print("coordinates are: ", answer)

            # Remove temp/temp.jpeg after processing
            os.remove('temp/temp.jpeg')
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
    elif prompt:
        answer = chatbot.continue_chat(prompt=prompt)
    else:
        return jsonify({'error': "No image_base64 or prompt provided in the request data."}), 400

    print("response: ",answer)
    return jsonify({'answer': answer})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
