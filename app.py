import torch
import hashlib
import transformers
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

from flask import Flask, render_template, request, jsonify, send_from_directory
import base64
import io
import time
import re
import threading
import atexit

# Create a Flask application
app = Flask(__name__, static_folder='static')

# Create necessary folders
os.makedirs('static/generated', exist_ok=True)
os.makedirs('static/characters', exist_ok=True)

# Global dict to track generation progress
generation_progress = {}
generation_results = {}

class ComicGenerator:
    def __init__(self):
        print("Loading Stable Diffusion model...")
       
        # Check if GPU is available and use it if possible
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
       
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None
        )
       
        # Apply optimizations for the detected hardware
        if self.device == "cuda":
            # GPU optimizations
            if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                self.pipe.enable_xformers_memory_efficient_attention()
            if hasattr(self.pipe, 'enable_model_cpu_offload'):
                self.pipe.enable_model_cpu_offload()
        else:
            # CPU optimizations that preserve quality
            self.pipe.enable_attention_slicing()
       
        # Apply better scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True
        )
       
        # Move to appropriate device
        self.pipe = self.pipe.to(self.device)
       
        # Character data storage
        self.character_concepts = {}
       
        print(f"Model loaded successfully on {self.device}!")
    
    def cleanup(self):
        """Clean up GPU resources when shutting down"""
        if self.device == "cuda":
            del self.pipe
            torch.cuda.empty_cache()
            print("Released GPU resources")
   
    def generate_character_embedding(self, character_name, character_description):
        """Generate stronger character concept with multiple reference images"""
        # Validate inputs
        if not character_name or not character_description:
            return False
            
        # Sanitize character name for filenames
        safe_name = re.sub(r'[^\w\s]', '', character_name).replace(' ', '_')
        
        # Create a detailed, consistent prompt for this character
        base_prompt = f"{character_name}: {character_description}, highly detailed, comic book style"
       
        # Create a directory for this character's reference images
        char_dir = f"static/characters/{safe_name}"
        os.makedirs(char_dir, exist_ok=True)
       
        # Generate multiple concept images with different poses for stronger character concept
        pose_prompts = [
            "portrait, face closeup, detailed facial features",
            "full body, standing pose, full view",
            "action pose, dynamic composition"
        ]
       
        for i, pose in enumerate(pose_prompts):
            # Use more steps and higher guidance for character reference images
            image = self.pipe(
                prompt=f"{base_prompt}, {pose}",
                negative_prompt="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limbs, missing limbs, floating limbs, mutated, multiple faces",
                num_inference_steps=30,  # More steps for higher quality reference
                guidance_scale=8.5,
                width=512,
                height=512
            ).images[0]
           
            # Save reference image
            image.save(f"{char_dir}/pose_{i}.png")
       
        # Extract key visual traits for consistency
        traits = self._extract_key_traits(character_description)
       
        # Store a strong concept description with physical traits
        self.character_concepts[character_name] = {
            'name': character_name,
            'base_prompt': base_prompt,
            'traits': traits,
            'ref_images': [f"{char_dir}/pose_{i}.png" for i in range(len(pose_prompts))]
        }
       
        # Save the main portrait as the primary reference
        portrait = Image.open(f"{char_dir}/pose_0.png")
        portrait.save(f"static/characters/{safe_name}.png")
       
        return True
   
    def _extract_key_traits(self, description):
        description = description.lower()
        traits = {}
    
    # Extract hair color/style with more variations and precision
        hair_patterns = [
        (["blonde", "blond hair", "golden hair", "yellow hair"], "blonde hair"),
        (["red hair", "ginger hair", "auburn hair", "copper hair"], "red hair"),
        (["brown hair", "brunette", "dark brown hair", "chestnut hair"], "brown hair"),
        (["black hair", "raven hair", "jet black hair", "dark hair"], "black hair"),
        (["white hair", "silver hair", "gray hair", "platinum hair"], "white/silver hair"),
        (["bald", "hairless", "shaved head", "no hair"], "bald")
        ]
    
        for patterns, trait in hair_patterns:
            for pattern in patterns:
                if pattern in description:
                    traits["hair"] = trait
                    break
            if "hair" in traits:
                break
    
    # Extract hairstyle if mentioned
        hairstyle_patterns = [
            (["long hair", "flowing hair"], "long hair"),
            (["short hair", "cropped hair"], "short hair"),
            (["curly hair", "wavy hair"], "curly/wavy hair"),
            (["straight hair"], "straight hair"),
            (["ponytail", "pigtail"], "hair in ponytail"),
            (["mohawk"], "mohawk"),
            (["afro"], "afro")
        ]
    
        for patterns, trait in hairstyle_patterns:
            for pattern in patterns:
                if pattern in description:
                    if "hair" in traits:
                        traits["hair"] += " with " + trait.split("hair ")[1] if "hair" in trait else trait
                    else:
                        traits["hair"] = trait
                    break
            if "hairstyle" in traits:
                break
    
    # Extract eye color
        eye_patterns = [
            (["blue eyes", "azure eyes"], "blue eyes"),
            (["green eyes", "emerald eyes"], "green eyes"),
            (["brown eyes", "hazel eyes"], "brown eyes"),
            (["black eyes", "dark eyes"], "black eyes"),
            (["red eyes", "crimson eyes"], "red eyes"),
            (["yellow eyes", "golden eyes", "amber eyes"], "yellow/golden eyes")
        ]
    
        for patterns, trait in eye_patterns:
            for pattern in patterns:
                if pattern in description:
                    traits["eyes"] = trait
                    break
            if "eyes" in traits:
                break
    
    # Extract clothing/costume details with more contextual awareness
        clothing_sections = []
        clothing_starters = ["wearing", "dressed in", "clothes", "outfit", "costume", "suit", "uniform", "cloak", "robe", "jacket"]
    
        for starter in clothing_starters:
            if starter in description:
                idx = description.find(starter)
            # Find the end of this clothing description (next sentence or comma)
                end_idx = description.find(".", idx)
                comma_idx = description.find(",", idx)
            
                if comma_idx != -1 and (end_idx == -1 or comma_idx < end_idx):
                    end_idx = comma_idx
            
                if end_idx == -1:
                    end_idx = len(description)
            
                clothing_section = description[idx:end_idx].strip()
                clothing_sections.append(clothing_section)
    
        if clothing_sections:
        # Use the longest clothing description for more detail
            longest_section = max(clothing_sections, key=len)
            traits["clothing"] = longest_section
    
    # Extract defining physical features with more precision
        physical_patterns = [
            (["tall", "towering", "giant", "huge"], "tall"),
            (["short", "small", "tiny", "little"], "short"),
            (["muscular", "strong", "athletic", "brawny", "buff"], "muscular"),
            (["slim", "thin", "slender", "lean"], "slim"),
            (["fat", "obese", "overweight", "heavy", "large"], "heavy-set"),
            (["athletic", "fit"], "athletic build")
        ]
    
        for patterns, trait in physical_patterns:
            for pattern in patterns:
                if pattern in description:
                    traits["build"] = trait
                    break
            if "build" in traits:
                break
    
    # Extract skin/color if mentioned with more variations
        skin_patterns = [
            (["blue skin", "azure skin", "sapphire skin"], "blue skin"),
            (["green skin", "emerald skin", "jade skin"], "green skin"),
            (["red skin", "crimson skin", "scarlet skin"], "red skin"),
            (["metal", "metallic", "steel", "iron", "chrome"], "metallic skin/body"),
            (["robot", "android", "mechanical", "automaton"], "robotic appearance"),
            (["alien", "extraterrestrial", "otherworldly"], "alien appearance"),
            (["pale skin", "fair skin", "light skin"], "pale skin"),
            (["tan skin", "tanned skin", "bronze skin"], "tanned skin"),
            (["dark skin", "brown skin", "ebony skin"], "dark skin")
        ]
    
        for patterns, trait in skin_patterns:
            for pattern in patterns:
                if pattern in description:
                    traits["skin"] = trait
                    break
            if "skin" in traits:
                break
    
    # Extract facial features like beard, mustache, scars
        facial_patterns = [
            (["beard", "bearded", "full beard"], "bearded"),
            (["mustache", "moustache"], "with mustache"),
            (["scar", "scarred", "facial scar"], "with facial scar"),
            (["glasses", "spectacles", "eyeglasses"], "wearing glasses"),
            (["mask", "masked", "face mask"], "wearing mask")
        ]
    
        for patterns, trait in facial_patterns:
            for pattern in patterns:
                if pattern in description:
                    traits["facial"] = trait
                    break
            if "facial" in traits:
                break
    
    # Extract distinctive accessories
        accessory_patterns = [
            (["hat", "cap", "helmet"], "wearing headgear"),
            (["sword", "blade", "katana"], "carrying sword"),
            (["gun", "pistol", "rifle"], "carrying firearm"),
            (["shield", "buckler"], "carrying shield"),
            (["staff", "wand", "scepter"], "carrying magical staff"),
            (["crown", "tiara", "diadem"], "wearing crown"),
            (["cape", "cloak", "mantle"], "wearing cape"),
            (["wings", "winged"], "with wings")
        ]
    
        for patterns, trait in accessory_patterns:
            for pattern in patterns:
                if pattern in description:
                    traits["accessory"] = trait
                    break
            if "accessory" in traits:
                break
    
        return traits

    def generate_panel(self, prompt, characters=None, width=512, height=512):
        start_time = time.time()
    
    # Build a more detailed base prompt for comic style
        full_prompt = prompt + ", comic book style, professional illustration, detailed art, crisp lines, vibrant colors"
        negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limbs, missing limbs, floating limbs, mutated, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, blurry, duplicate, multiplied, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, poorly drawn, gross proportions, text, watermark"
    
    # Add detailed character descriptions for consistency
        if characters:
            character_descriptions = []
            characters_in_prompt = []
        
        # Check which characters are mentioned in the prompt
            for character in characters:
                if character.lower() in prompt.lower():
                    characters_in_prompt.append(character)
        
        # If no characters are explicitly mentioned, use all selected characters
            if not characters_in_prompt and characters:
                characters_in_prompt = characters
        
            for character in characters_in_prompt:
                if character in self.character_concepts:
                # Extract detailed traits for consistent appearance
                    char_info = self.character_concepts[character]
                
                # Create a detailed character description with all traits for consistent appearance
                    trait_descriptions = []
                    for category, trait in char_info['traits'].items():
                        trait_descriptions.append(f"{trait}")
                
                    traits_str = ", ".join(trait_descriptions)
                
                # Add character to the prompt with detailed description
                    char_prompt = f"{character} ({traits_str})"
                    character_descriptions.append(char_prompt)
        
        # Integrate characters into the prompt
            if character_descriptions:
                if "featuring" in prompt.lower() or "with" in prompt.lower():
                # Characters might already be mentioned in a specific way
                    full_prompt = f"{full_prompt}"
                else:
                # Add characters explicitly
                    full_prompt = f"{full_prompt} featuring {', '.join(character_descriptions)}"
    
        print(f"Generating panel with prompt: {full_prompt}")
    
    # Run inference with better quality settings
        image = self.pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,  # More steps for better quality
            guidance_scale=7.5,
            width=width,
            height=height
        ).images[0]
    
    # Add comic border
        image = self.add_comic_border(image)
    
        print(f"Panel generation took {time.time() - start_time:.2f} seconds")
        return image
   
    def generate_story_panels(self, story_prompt, num_panels=3, characters=None, session_id=None):
        """Generate a sequence of panels based on a story prompt with progress tracking"""
        if not story_prompt:
            return []
           
        # Break down story into better panel prompts with narrative structure
        panel_prompts = self.story_to_panel_prompts(story_prompt, num_panels)
       
        panels = []
        for i, panel_prompt in enumerate(panel_prompts):
            print(f"Generating panel {i+1}/{len(panel_prompts)}...")
           
            # Update progress if session_id provided
            if session_id:
                generation_progress[session_id] = int((i / len(panel_prompts)) * 100)
           
            panel = self.generate_panel(panel_prompt, characters)
            panels.append((panel,panel_prompt))
           
            # Update progress after completion
            if session_id:
                generation_progress[session_id] = int(((i + 1) / len(panel_prompts)) * 100)
           
        return panels
   
    def story_to_panel_prompts(self, story_prompt, num_panels):
        """Break down a story prompt into individual panel descriptions with better narrative structure"""
        # If panel descriptions are explicitly provided, parse and use those
        if "Panel" in story_prompt or "panel" in story_prompt:
            panels = re.split(r'Panel \d+:|panel \d+:', story_prompt)
            panels = [p.strip() for p in panels if p.strip()]
           
            # If we have more specified panels than requested, truncate
            if len(panels) > num_panels:
                panels = panels[:num_panels]
               
            # If we have fewer, use a narrative structure to complete
            if len(panels) < num_panels:
                last_panel = panels[-1] if panels else story_prompt
                remaining = num_panels - len(panels)
               
                # Add resolution panels as needed
                for i in range(remaining):
                    if i == remaining - 1:
                        panels.append(f"{last_panel}, conclusion, resolution of the story")
                    else:
                        panels.append(f"{last_panel}, continuation, progressing the action")
           
            return panels
        
        # Otherwise, create a proper narrative arc
        if num_panels <= 1:
            return [f"{story_prompt}, key dramatic moment, comic book style"]
        
        # Create a proper narrative structure
        narrative_beats = [
            f"{story_prompt}, establishing shot, introduction, beginning of the story",
            f"{story_prompt}, character introduction, setting the scene",
            f"{story_prompt}, rising action, conflict emerges",
            f"{story_prompt}, confrontation, peak action moment",
            f"{story_prompt}, climax, most dramatic moment",
            f"{story_prompt}, resolution, conclusion of the story"
        ]
        
        # For smaller panel counts, select the most important narrative beats
        if num_panels == 2:
            return [narrative_beats[0], narrative_beats[-1]]
        elif num_panels == 3:
            return [narrative_beats[0], narrative_beats[2], narrative_beats[-1]]
        elif num_panels == 4:
            return [narrative_beats[0], narrative_beats[1], narrative_beats[3], narrative_beats[-1]]
        else:
            # Use as many as specified, up to our prepared narrative beats
            return narrative_beats[:min(num_panels, len(narrative_beats))]
    
    def add_comic_border(self, image):
        """Add a comic-style border to the image"""
        border_size = 5
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Draw black border
        draw.rectangle([(0, 0), (width-1, height-1)], outline="black", width=border_size)
        
        return image

# Initialize the comic generator
comic_generator = ComicGenerator()

# Serve the main page
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/generate_character', methods=['POST'])
def generate_character():
    data = request.json
    character_name = data.get('name', '')
    character_description = data.get('description', '')
    
    if not character_name or not character_description:
        return jsonify({'success': False, 'error': 'Name and description required'}), 400
    
    try:
        success = comic_generator.generate_character_embedding(character_name, character_description)
        
        # Get all available characters
        characters = list(comic_generator.character_concepts.keys())
        
        return jsonify({
            'success': success,
            'message': f"Character {character_name} created",
            'character_image': f"/static/characters/{character_name.replace(' ', '_')}.png",
            'all_characters': characters
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/characters', methods=['GET'])
def get_characters():
    # Get all available characters
    characters = list(comic_generator.character_concepts.keys())
    return jsonify({
        'characters': characters
    })

# @app.route('/generate_comic', methods=['POST'])
# def generate_comic():
#     data = request.json
#     story_prompt = data.get('prompt', '')
#     selected_characters = data.get('characters', [])
#     num_panels = min(int(data.get('num_panels', 3)), 12)  # Limit to 6 panels max
    
#     if not story_prompt:
#         return jsonify({'success': False, 'error': 'Story prompt required'}), 400
    
#     try:
#         # Create a session ID for this generation
#         session_id = str(int(time.time()))
#         generation_progress[session_id] = 0
        
#         # Start a background thread for generation
#         def generate_in_background():
#             try:
#                 # Generate panels
#                 panels = comic_generator.generate_story_panels(
#                     story_prompt, 
#                     num_panels=num_panels,
#                     characters=selected_characters,
#                     session_id=session_id
#                 )
                
#                 # Convert panels to base64 and save to disk
#                 panel_images = []
#                 for i, (panel, panel_prompt) in enumerate(panels):
#                     buffered = io.BytesIO()
#                     panel.save(buffered, format="PNG")
#                     img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
#                     panel_filename = f"panel_{session_id}_{i}.png"
#                     panel.save(f"static/generated/{panel_filename}")
                    
#                     panel_images.append({
#                         'data': img_str,
#                         'filename': panel_filename,
#                         'description': panel_prompt[:100] + '...' if len(panel_prompt) > 100 else panel_prompt
#                     })
                
#                 # Store the results
#                 generation_results[session_id] = {
#                     'success': True,
#                     'panels': panel_images,
#                     'message': f"Generated {len(panels)} comic panels"
#                 }
                
#                 # Mark as complete
#                 generation_progress[session_id] = 100
                
#             except Exception as e:
#                 import traceback
#                 traceback.print_exc()
#                 generation_results[session_id] = {
#                     'success': False,
#                     'error': str(e)
#                 }
#                 generation_progress[session_id] = -1  # Error state
        
#         thread = threading.Thread(target=generate_in_background)
#         thread.daemon = True
#         thread.start()
        
#         return jsonify({
#             'success': True,
#             'session_id': session_id,
#             'message': f"Generation started for {num_panels} panels"
#         })
        
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/check_progress/<session_id>', methods=['GET'])
def check_progress(session_id):
    """Check the progress of a comic generation"""
    if session_id == "undefined" or session_id not in generation_progress:
        return jsonify({'progress': -1, 'complete': True, 'error': 'Invalid session ID'}), 400
    
    progress = generation_progress.get(session_id, 0)
    
    # If progress is 100%, return the results too
    if progress == 100 and session_id in generation_results:
        return jsonify({
            'progress': progress,
            'complete': True,
            'results': generation_results[session_id]
        })
    elif progress == -1 and session_id in generation_results:
        # Error occurred
        return jsonify({
            'progress': -1,
            'complete': True,
            'error': generation_results[session_id].get('error', 'Unknown error')
        })
    else:
        return jsonify({
            'progress': progress,
            'complete': False
        })

# Clean up old results periodically (optional)
def cleanup_old_results():
    """Remove old generation results to free memory"""
    current_time = time.time()
    for session_id in list(generation_results.keys()):
        # Remove results older than 1 hour
        if float(session_id) < current_time - 3600:
            if session_id in generation_results:
                del generation_results[session_id]
            if session_id in generation_progress:
                del generation_progress[session_id]
def generate_panel_hash(prompt, characters):
    """Generate a unique hash for a given prompt and character list."""
    input_str = prompt + "".join(sorted(characters))
    return hashlib.sha256(input_str.encode()).hexdigest()

def add_speech_bubble(image_path, text, position=(50, 50), bubble_size=(250, 100)):
    """Add a speech bubble to the comic panel."""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Define bubble parameters
    x, y = position
    w, h = bubble_size
    bubble_color = "white"
    outline_color = "black"

    # Draw the speech bubble (rounded rectangle)
    draw.rounded_rectangle([x, y, x + w, y + h], fill=bubble_color, outline=outline_color, radius=20)

    # Draw speech tail (small triangle)
    tail_points = [(x + w - 20, y + h), (x + w, y + h + 20), (x + w - 40, y + h)]
    draw.polygon(tail_points, fill=bubble_color, outline=outline_color)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Add text
    draw.text((x + 15, y + 15), text, fill="black", font=font)

    # Save modified image
    modified_image_path = image_path.replace(".png", "_bubble.png")
    image.save(modified_image_path)
    return modified_image_path

comic_generator = ComicGenerator()
@app.route('/generate_comic', methods=['POST'])
def generate_comic():
    data = request.json
    story_prompt = data.get('prompt', '')
    selected_characters = data.get('characters', [])
    num_panels = min(int(data.get('num_panels', 3)), 6)
    
    if not story_prompt:
        return jsonify({'success': False, 'error': 'Story prompt required'}), 400
    
    # Generate hash for prompt + characters
    panel_hash = generate_panel_hash(story_prompt, selected_characters)
    
    # Check if previously generated
    stored_panels = []
    for i in range(num_panels):
        panel_filename = f"panel_{panel_hash}_{i}.png"
        panel_path = os.path.join("static/generated", panel_filename)
        if os.path.exists(panel_path):
            stored_panels.append({'filename': panel_filename, 'description': f"Panel {i + 1}"})
    
    if len(stored_panels) == num_panels:
        return jsonify({'success': True, 'panels': stored_panels, 'message': 'Loaded from cache'})
    
    # If not stored, generate new panels
    session_id = str(int(time.time()))
    generation_progress[session_id] = 0

    
    def generate_in_background():
        try:
            panels = comic_generator.generate_story_panels(story_prompt, num_panels, selected_characters, session_id)
            panel_images = []
            for i, (panel, panel_prompt) in enumerate(panels):
                panel_filename = f"panel_{panel_hash}_{i}.png"
                panel_path = os.path.join("static/generated", panel_filename)
                panel.save(panel_path)
                panel_images.append({'filename': panel_filename, 'description': panel_prompt})
            
            generation_results[session_id] = {'success': True, 'panels': panel_images, 'message': 'Panels generated'}
            generation_progress[session_id] = 100
        except Exception as e:
            generation_results[session_id] = {'success': False, 'error': str(e)}
            generation_progress[session_id] = -1

        thread = threading.Thread(target=generate_in_background)
        thread.daemon = True
        thread.start()

        return jsonify({'success': True, 'session_id': session_id, 'message': 'Generation started'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)