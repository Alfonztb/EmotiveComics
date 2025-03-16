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
                num_inference_steps=40,  # More steps for higher quality reference
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

@app.route('/generate_comic', methods=['POST'])
def generate_comic():
    data = request.json
    story_prompt = data.get('prompt', '')
    selected_characters = data.get('characters', [])
    num_panels = min(int(data.get('num_panels', 3)), 12)  # Limit to 12 panels max
    
    if not story_prompt:
        return jsonify({'success': False, 'error': 'Story prompt required'}), 400
    
    try:
        # Create a session ID for this generation
        session_id = str(int(time.time()))
        generation_progress[session_id] = 0
        
        # Start a background thread for generation
        def generate_in_background():
            try:
                # Generate panels
                panels = comic_generator.generate_story_panels(
                    story_prompt, 
                    num_panels=num_panels,
                    characters=selected_characters,
                    session_id=session_id
                )
                
                # Convert panels to base64 and save to disk
                panel_images = []
                for i, (panel, panel_prompt) in enumerate(panels):
                    buffered = io.BytesIO()
                    panel.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    panel_filename = f"panel_{session_id}_{i}.png"
                    panel.save(f"static/generated/{panel_filename}")
                    
                    panel_images.append({
                        'data': img_str,
                        'filename': panel_filename,
                        'description': panel_prompt[:100] + '...' if len(panel_prompt) > 100 else panel_prompt
                    })
                
                # Store the results
                generation_results[session_id] = {
                    'success': True,
                    'panels': panel_images,
                    'message': f"Generated {len(panels)} comic panels"
                }
                
                # Mark as complete
                generation_progress[session_id] = 100
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                generation_results[session_id] = {
                    'success': False,
                    'error': str(e)
                }
                generation_progress[session_id] = -1  # Error state
        
        thread = threading.Thread(target=generate_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': f"Generation started for {num_panels} panels"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

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

def create_dynamic_comic_layout(panels, dialogues=None, max_width=1200, margin=20):
    """
    Creates a dynamic comic layout with panels sized according to their content and importance.
    
    Args:
        panels: List of (image, prompt) tuples containing panel images and their descriptions
        dialogues: Dictionary mapping panel indices to character dialogues
        max_width: Maximum width of the overall comic page
        margin: Margin between panels
        
    Returns:
        PIL Image of the complete comic with all panels arranged
    """
    if not panels:
        return None
    
    # Analyze prompts and dialogues to determine panel importance
    panel_importance = []
    for i, (_, prompt) in enumerate(panels):
        # Base importance on prompt length, dialogue amount, and keywords
        importance = len(prompt) / 100  # Longer prompts might need more space
        
        # Add importance for panels with dialogue
        if dialogues and i in dialogues:
            importance += len(dialogues[i]) * 0.5  # More characters talking = more important
            dialogue_length = sum(len(text) for text in dialogues[i].values())
            importance += dialogue_length / 200  # Longer dialogues need more space
        
        # Check for action keywords that might indicate an important scene
        action_keywords = ["fight", "explosion", "confrontation", "climax", "reveal"]
        for keyword in action_keywords:
            if keyword in prompt.lower():
                importance += 0.5
                
        panel_importance.append(importance)
    
    # Normalize importance scores to a range we can use for sizing
    max_imp = max(panel_importance) if panel_importance else 1
    min_imp = min(panel_importance) if panel_importance else 0
    imp_range = max(max_imp - min_imp, 1)  # Avoid division by zero
    
    normalized_importance = [
        0.5 + ((imp - min_imp) / imp_range) * 0.5  # Range from 0.5 to 1.0
        for imp in panel_importance
    ]
    
    # Calculate ideal panel dimensions based on importance
    num_panels = len(panels)
    
    # Determine layout (how many rows and columns)
    if num_panels <= 2:
        num_cols = num_panels
        num_rows = 1
    elif num_panels <= 4:
        num_cols = 2
        num_rows = (num_panels + 1) // 2
    elif num_panels <= 6:
        num_cols = 3
        num_rows = (num_panels + 2) // 3
    else:
        num_cols = 3
        num_rows = (num_panels + 2) // 3
    
    # Adjust based on panel importance - important panels can span multiple grid cells
    # This is a simplified approach; a more advanced layout algorithm would be needed
    # for truly dynamic layouts
    
    # Calculate base panel width and height
    base_panel_width = (max_width - (margin * (num_cols + 1))) // num_cols
    
    # First, prepare all panels by adding speech bubbles if needed
    prepared_panels = []
    for i, (panel_img, _) in enumerate(panels):
        # Process with speech bubbles if needed
        if dialogues and i in dialogues:
            # Get original panel dimensions for aspect ratio calculation
            panel_width, panel_height = panel_img.size
            aspect_ratio = panel_height / panel_width
            
            # Detect character positions
            panel_path = f"temp_panel_{i}.png"
            panel_img.save(panel_path)
            char_positions = detect_character_positions(panel_path, comic_generator.character_concepts)
            os.remove(panel_path)  # Clean up temp file
            
            # Create a copy for speech bubbles
            panel_with_speech = panel_img.copy()
            
            # Add speech bubbles directly to the image
            for char_name, speech_text in dialogues[i].items():
                panel_with_speech = add_speech_bubble_to_image(
                    panel_with_speech,
                    speech_text,
                    character_name=char_name,
                    characters=char_positions
                )
            
            prepared_panels.append((panel_with_speech, aspect_ratio, normalized_importance[i]))
        else:
            panel_width, panel_height = panel_img.size
            aspect_ratio = panel_height / panel_width
            prepared_panels.append((panel_img, aspect_ratio, normalized_importance[i]))
    
    # Calculate layout grid - which panels go where and what size they should be
    grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    panel_sizes = {}  # Maps (row, col) to (width, height) in grid units
    
    # Sort panels by importance, placing more important panels first
    panel_indices = list(range(len(prepared_panels)))
    panel_indices.sort(key=lambda i: prepared_panels[i][2], reverse=True)
    
    # Place panels on the grid
    for idx in panel_indices:
        panel_img, aspect_ratio, importance = prepared_panels[idx]
        
        # Determine if this panel should span multiple cells based on importance
        span_h = 1
        span_v = 1
        
        if importance > 0.8 and num_panels > 3:
            # Very important panels can be larger
            if idx == panel_indices[0]:  # Most important panel
                if num_cols >= 2:
                    span_h = 2  # Span two columns for the most important panel
                if num_rows >= 2 and num_panels > 4:
                    span_v = 2  # Potentially span two rows for very important panels
        
        # Find a place on the grid for this panel
        placed = False
        for row in range(num_rows - span_v + 1):
            for col in range(num_cols - span_h + 1):
                # Check if this spot and needed spans are available
                spot_available = True
                for r in range(span_v):
                    for c in range(span_h):
                        if grid[row + r][col + c] is not None:
                            spot_available = False
                            break
                    if not spot_available:
                        break
                
                if spot_available:
                    # Place the panel here
                    for r in range(span_v):
                        for c in range(span_h):
                            grid[row + r][col + c] = idx
                    
                    # Store the panel's size in grid units
                    panel_sizes[(row, col)] = (span_h, span_v)
                    placed = True
                    break
            if placed:
                break
        
        # If we couldn't place with spans, try again with a 1x1 panel
        if not placed:
            for row in range(num_rows):
                for col in range(num_cols):
                    if grid[row][col] is None:
                        grid[row][col] = idx
                        panel_sizes[(row, col)] = (1, 1)
                        placed = True
                        break
                if placed:
                    break
    
    # Calculate actual pixel dimensions for each panel
    layout_map = {}  # Maps panel index to (x, y, width, height) in the final image
    
    # Calculate row heights based on content
    row_heights = [0] * num_rows
    for row in range(num_rows):
        for col in range(num_cols):
            if grid[row][col] is not None:
                idx = grid[row][col]
                origin_row, origin_col = None, None
                
                # Find the origin cell for this panel
                for r in range(row + 1):
                    for c in range(num_cols):
                        if grid[r][c] == idx:
                            origin_row, origin_col = r, c
                            break
                    if origin_row is not None:
                        break
                
                # If this is the origin cell, consider its height
                if (row, col) == (origin_row, origin_col):
                    span_h, span_v = panel_sizes[(row, col)]
                    _, aspect_ratio, _ = prepared_panels[idx]
                    
                    # Calculate panel width based on column span
                    panel_width = span_h * base_panel_width + (span_h - 1) * margin
                    
                    # Calculate panel height based on aspect ratio
                    panel_height = int(panel_width * aspect_ratio)
                    
                    # Update row height if this panel requires more height
                    row_contribution = panel_height // span_v  # Height contribution to each row
                    row_heights[row] = max(row_heights[row], row_contribution)
    
    # Ensure minimum height for each row
    min_row_height = 150
    row_heights = [max(h, min_row_height) for h in row_heights]
    
    # Calculate total height with margins
    total_height = sum(row_heights) + (num_rows + 1) * margin
    
    # Create the final comic image
    comic_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    
    # Place each panel on the comic
    for row in range(num_rows):
        for col in range(num_cols):
            if grid[row][col] is not None:
                idx = grid[row][col]
                origin_row, origin_col = None, None
                
                # Skip if this is not the origin cell of the panel
                for r in range(row):
                    for c in range(num_cols):
                        if grid[r][c] == idx:
                            origin_row, origin_col = r, c
                            break
                    if origin_row is not None:
                        break
                
                for c in range(col):
                    if grid[row][c] == idx:
                        origin_row, origin_col = row, c
                        break
                
                if origin_row is not None or origin_col is not None:
                    continue
                
                # This is the origin cell for this panel
                panel_img, _, _ = prepared_panels[idx]
                span_h, span_v = panel_sizes[(row, col)]
                
                # Calculate actual width and height for this panel
                panel_width = span_h * base_panel_width + (span_h - 1) * margin
                
                # Calculate height based on the row heights
                panel_height = sum(row_heights[row:row+span_v]) + (span_v - 1) * margin
                
                # Resize the panel image to fit the allocated space
                panel_img = panel_img.resize((panel_width, panel_height), Image.LANCZOS)
                
                # Calculate x and y position in the final image
                x = margin + col * (base_panel_width + margin)
                y = margin + sum(row_heights[:row]) + row * margin
                
                # Add a border around the panel
                draw = ImageDraw.Draw(panel_img)
                draw.rectangle([(0, 0), (panel_width-1, panel_height-1)], outline="black", width=3)
                
                # Place the panel
                comic_image.paste(panel_img, (x, y))
                
                # Store the layout information
                layout_map[idx] = (x, y, panel_width, panel_height)
    
    return comic_image, layout_map

def add_speech_bubble_to_image(image, text, character_name=None, characters=None):
    """
    Add a speech bubble directly to an image (not using file paths).
    
    Args:
        image: PIL Image object
        text: Text to put in the speech bubble
        character_name: Name of the character speaking (for placement)
        characters: Dictionary of character information including positions
        
    Returns:
        PIL Image with speech bubble added
    """
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    # Choose font based on text length for better readability
    font_size = min(30, max(16, int(400 / (len(text) / 10)))) if len(text) > 0 else 30
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fallback to default font if arial not available
        font = ImageFont.load_default()

    # Handle text wrapping for better speech bubble layout
    max_chars_per_line = min(30, max(10, int(img_width / 15)))  # Adjust based on image width
    wrapped_text = textwrap.fill(text, width=max_chars_per_line)
    
    # Calculate text dimensions with wrapping
    text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Add padding around text for speech bubble
    bubble_padding = 20
    bubble_width = text_width + bubble_padding * 2
    bubble_height = text_height + bubble_padding * 2
    
    # Determine character position in the image for better speech bubble placement
    char_position = "unknown"
    tail_points = []
    
    if character_name and characters and character_name in characters:
        # Use character position information if available
        char_info = characters.get(character_name, {})
        char_position = char_info.get("position", "unknown")
        
        # If we have facial coordinates, use those for precise placement
        if "face_coords" in char_info:
            face_x, face_y = char_info["face_coords"]
            
            # Place bubble based on character face position
            if face_y < img_height / 3:  # Character in top third
                bubble_x = max(10, min(face_x - bubble_width/2, img_width - bubble_width - 10))
                bubble_y = min(face_y + 50, img_height - bubble_height - 10)
                tail_points = [(bubble_x + bubble_width/2, bubble_y), (face_x, face_y + 10)]
            else:  # Character elsewhere
                bubble_x = max(10, min(face_x - bubble_width/2, img_width - bubble_width - 10))
                bubble_y = max(10, face_y - bubble_height - 30)
                tail_points = [(bubble_x + bubble_width/2, bubble_y + bubble_height), (face_x, face_y - 10)]
    
    # Default positioning if character position unknown
    if char_position == "unknown" or not tail_points:
        # Analyze image to find less busy areas for bubble placement
        # This is a simplified approach - a real implementation would use 
        # image analysis to find suitable areas
        bubble_x = (img_width - bubble_width) // 2
        bubble_y = 20
    
    # Ensure bubble stays within image bounds
    bubble_x = max(10, min(bubble_x, img_width - bubble_width - 10))
    bubble_y = max(10, min(bubble_y, img_height - bubble_height - 10))
    
    # Create a new image that might be larger to accommodate the speech bubble
    padding_needed = max(0, bubble_y + bubble_height + 10 - img_height)
    if padding_needed > 0:
        new_height = img_height + padding_needed
        new_image = Image.new("RGB", (img_width, new_height), (255, 255, 255))
        new_image.paste(image, (0, 0))
        image = new_image
        draw = ImageDraw.Draw(image)
    
    # Draw the speech bubble
    draw.rounded_rectangle(
        [bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + bubble_height],
        radius=10,
        fill="white",
        outline="black",
        width=2
    )
    
    # Draw tail connecting bubble to character if we have valid tail points
    if tail_points:
        draw.line(tail_points, fill="black", width=2)
    
    # Draw text inside bubble with proper alignment
    lines = wrapped_text.split('\n')
    y_text = bubble_y + bubble_padding
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        x_text = bubble_x + (bubble_width - line_width) // 2  # Center text
        draw.text((x_text, y_text), line, fill="black", font=font)
        y_text += font_size + 2  # Move to next line
    
    return image

# Main function to generate a comic with dynamic panel layout
def generate_dynamic_comic(story_prompt, characters, dialogues=None, num_panels=3):
    """
    Generate a comic with dynamic panel layout and speech bubbles.
    
    Args:
        story_prompt: The story prompt for generating panels
        characters: List of character names to include
        dialogues: Dictionary mapping panel indices to character dialogues
                   {panel_idx: {character_name: "dialogue text"}}
        num_panels: Base number of panels to generate
        
    Returns:
        Path to the final comic image with all panels arranged
    """
    # First, generate individual panels
    panels = comic_generator.generate_story_panels(story_prompt, num_panels, characters)
    
    # Create the dynamic layout
    comic_image, layout_map = create_dynamic_comic_layout(panels, dialogues, max_width=1200)
    
    # Save the final comic
    timestamp = int(time.time())
    comic_path = f"static/generated/comic_{timestamp}.png"
    comic_image.save(comic_path)
    
    return comic_path, layout_map

# Improved character position detection
def detect_character_positions(image_path, character_concepts):
    """
    Analyzes the image to detect character positions for better speech bubble placement.
    Uses color distribution and edge detection for improved accuracy.
    
    Args:
        image_path: Path to the panel image
        character_concepts: Dictionary of character concepts with visual traits
        
    Returns:
        Dictionary mapping character names to their estimated positions in the image
    """
    # Open the image
    image = Image.open(image_path)
    width, height = image.size
    
    # Convert to numpy array for analysis
    img_array = np.array(image)
    
    # Use edge detection to find areas with likely character outlines
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray_img = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        gray_img = img_array.astype(np.uint8)
    
    # Apply Sobel edge detection
    sobel_x = np.abs(np.gradient(gray_img, axis=1))
    sobel_y = np.abs(np.gradient(gray_img, axis=0))
    edge_strength = sobel_x + sobel_y
    
    # Divide the image into a grid for analysis
    grid_size = 4  # Increased from 3 for better granularity
    grid_width = width // grid_size
    grid_height = height // grid_size
    
    # Score each grid cell based on edge density and color variance
    grid_scores = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Extract grid cell
            y_start, y_end = j*grid_height, min((j+1)*grid_height, height)
            x_start, x_end = i*grid_width, min((i+1)*grid_width, width)
            
            cell = img_array[y_start:y_end, x_start:x_end]
            cell_edges = edge_strength[y_start:y_end, x_start:x_end]
            
            # Calculate edge density (higher values indicate potential character outlines)
            edge_density = np.mean(cell_edges)
            
            # Calculate color variance as indicator of detail
            if len(cell.shape) == 3:  # Color image
                color_variance = np.var(cell, axis=(0, 1)).mean()
            else:  # Grayscale
                color_variance = np.var(cell)
            
            # Combined score favoring areas with high edge density and moderate color variance
            grid_scores[j, i] = (edge_density * 0.7) + (color_variance * 0.3)
    
    # Normalize grid scores
    if np.max(grid_scores) > 0:
        grid_scores = grid_scores / np.max(grid_scores)
    
    # Assign characters to grid cells with highest scores
    char_positions = {}
    assigned_cells = set()
    
    # Sort characters by importance or appearance in the story
    char_names = list(character_concepts.keys())
    
    # Find top scoring cells for each character, avoiding adjacent cells for different characters
    for char_name in char_names:
        # Find best unassigned cell
        best_score = -1
        best_cell = (0, 0)
        
        for j in range(grid_size):
            for i in range(grid_size):
                # Skip if cell is already assigned or adjacent to assigned cell
                if (j, i) in assigned_cells:
                    continue
                    
                # Check for adjacent assigned cells to avoid character overlap
                adjacent_occupied = False
                for dj in [-1, 0, 1]:
                    for di in [-1, 0, 1]:
                        if (j+dj, i+di) in assigned_cells and (dj != 0 or di != 0):
                            adjacent_occupied = True
                            break
                    if adjacent_occupied:
                        break
                
                # Skip if adjacent to another character
                if adjacent_occupied and len(assigned_cells) < grid_size:  # Skip adjacency check if running out of cells
                    continue
                    
                if grid_scores[j, i] > best_score:
                    best_score = grid_scores[j, i]
                    best_cell = (j, i)
        
        # Calculate face coordinates within the cell (aim for upper half where faces usually are)
        face_y = int((best_cell[0] + 0.3) * grid_height)  # Bias toward upper part of cell for face
        face_x = int((best_cell[1] + 0.5) * grid_width)
        
        # Determine position label
        position_y = "top" if best_cell[0] < grid_size/3 else "middle" if best_cell[0] < 2*grid_size/3 else "bottom"
        position_x = "left" if best_cell[1] < grid_size/3 else "center" if best_cell[1] < 2*grid_size/3 else "right"
        position = f"{position_y}-{position_x}"
        
        # Add to character positions
        char_positions[char_name] = {
            "position": position,
            "face_coords": (face_x, face_y),
            "score": best_score
        }
        
        # Mark cell as assigned
        assigned_cells.add(best_cell)
    
    return char_positions

comic_generator = ComicGenerator()
@app.route('/generate_dynamic_comic', methods=['POST'])
def generate_dynamic_comic_route():
    data = request.json
    story_prompt = data.get('prompt', '')
    selected_characters = data.get('characters', [])
    num_panels = min(int(data.get('num_panels', 3)), 12)  # Allow more panels for dynamic layout
    dialogues = data.get('dialogues', {})  # {panel_idx: {character_name: "dialogue"}}
    
    # Convert dialogue panel indices from strings to integers
    formatted_dialogues = {}
    for panel_idx, char_dialogues in dialogues.items():
        formatted_dialogues[int(panel_idx)] = char_dialogues
    
    if not story_prompt:
        return jsonify({'success': False, 'error': 'Story prompt required'}), 400
    
    session_id = str(int(time.time()))
    generation_progress[session_id] = 0
    
    def generate_in_background():
        try:
            # Generate the comic with dynamic layout
            comic_path, layout_map = generate_dynamic_comic(
                story_prompt, 
                selected_characters, 
                formatted_dialogues,
                num_panels
            )
            
            # Extract just the filename from the path
            comic_filename = os.path.basename(comic_path)
            
            generation_results[session_id] = {
                'success': True, 
                'comic_path': comic_filename,
                'layout': layout_map,
                'message': 'Comic generated with dynamic layout'
            }
            generation_progress[session_id] = 100
        except Exception as e:
            import traceback
            traceback.print_exc()
            generation_results[session_id] = {
                'success': False, 
                'error': str(e)
            }
            generation_progress[session_id] = -1
    
    thread = threading.Thread(target=generate_in_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True, 
        'session_id': session_id, 
        'message': 'Dynamic comic generation started'
    })
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)