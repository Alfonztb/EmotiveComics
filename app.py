import torch
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
    
    def generate_character_embedding(self, character_name, character_description):
        """Generate stronger character concept with multiple reference images"""
        # Create a detailed, consistent prompt for this character
        base_prompt = f"{character_name}: {character_description}, highly detailed, comic book style"
        
        # Create a directory for this character's reference images
        char_dir = f"static/characters/{character_name.replace(' ', '_')}"
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
        portrait.save(f"static/characters/{character_name.replace(' ', '_')}.png")
        
        return True
    
    def _extract_key_traits(self, description):
        """Extract key physical traits for character consistency"""
        traits = {}
        
        # Extract hair color/style
        hair_patterns = [
            ("blonde", "blonde hair"), 
            ("blond hair", "blonde hair"),
            ("red hair", "red hair"),
            ("brunette", "brown hair"),
            ("black hair", "black hair"),
            ("white hair", "white hair"),
            ("bald", "bald")
        ]
        
        for pattern, trait in hair_patterns:
            if pattern in description.lower():
                traits["hair"] = trait
                break
        
        # Extract clothing/costume details
        clothing_patterns = ["wearing", "dressed in", "clothes", "outfit", "costume", "suit"]
        for pattern in clothing_patterns:
            if pattern in description.lower():
                idx = description.lower().find(pattern)
                # Extract clothing description from around this pattern
                # Find the end of the sentence or a reasonable chunk
                end_idx = description.find(".", idx)
                if end_idx == -1 or end_idx - idx > 100:
                    end_idx = min(idx + 100, len(description))
                
                clothing_desc = description[idx:end_idx]
                traits["clothing"] = clothing_desc
                break
        
        # Extract defining physical features
        physical_patterns = ["tall", "short", "muscular", "slim", "thin", "fat", "strong", "big"]
        for pattern in physical_patterns:
            if pattern in description.lower():
                if "body" not in traits:
                    traits["body"] = pattern
        
        # Extract skin/color if mentioned
        color_patterns = [
            ("blue skin", "blue skin"),
            ("green skin", "green skin"),
            ("red skin", "red skin"),
            ("metal", "metallic skin"),
            ("robot", "robotic appearance"),
            ("alien", "alien appearance")
        ]
        
        for pattern, trait in color_patterns:
            if pattern in description.lower():
                traits["appearance"] = trait
                break
        
        return traits
    
    def generate_panel(self, prompt, characters=None, width=512, height=512):
        """Generate a higher quality comic panel with consistent characters"""
        start_time = time.time()
        
        # Build a more detailed base prompt for comic style
        full_prompt = prompt + ", comic book style, professional illustration, detailed art, crisp lines, vibrant colors"
        negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limbs, missing limbs, floating limbs, mutated, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, blurry, duplicate, multiplied, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, poorly drawn, gross proportions, text, watermark"
        
        # Add detailed character descriptions for consistency
        if characters:
            character_descriptions = []
            for character in characters:
                if character in self.character_concepts:
                    # Extract detailed traits for consistent appearance
                    char_info = self.character_concepts[character]
                    traits_str = ", ".join([f"{k}: {v}" for k, v in char_info['traits'].items()])
                    character_descriptions.append(f"{char_info['base_prompt']}, {traits_str}")
            
            if character_descriptions:
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
            panels.append(panel)
            
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
    return send_from_directory('static', 'index.html')

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

@app.route('/get_characters', methods=['GET'])
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
    num_panels = min(int(data.get('num_panels', 3)), 6)  # Limit to 6 panels max
    
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
                for i, panel in enumerate(panels):
                    buffered = io.BytesIO()
                    panel.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    panel_filename = f"panel_{session_id}_{i}.png"
                    panel.save(f"static/generated/{panel_filename}")
                    
                    panel_images.append({
                        'data': img_str,
                        'filename': panel_filename
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)