# Libraries for building GUI 
import tkinter as tk
import customtkinter as ctk 

# Machine Learning libraries 
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Libraries for processing image 
from PIL import ImageTk

# private modules 
from authtoken import auth_token


# Create app user interface
app = tk.Tk()
app.geometry("532x632")
app.title("Text to Image app")
app.configure(bg='black')
ctk.set_appearance_mode("dark") 

# Create input box on the user interface 
prompt = ctk.CTkEntry(height=40, width=512, text_font=("Arial", 15), text_color="white", fg_color="black") 
prompt.place(x=10, y=10)

# Create a placeholder to show the generated image
img_placeholder = ctk.CTkLabel(height=512, width=512, text="")
img_placeholder.place(x=10, y=110)

# Download stable diffusion model from hugging face 
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
stable_diffusion_model = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
stable_diffusion_model.to(device) 

# Generate image from test 
def generate(): 
    """ This function generate image from a text with stable diffusion"""
    with autocast(device): 
        image = stable_diffusion_model(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    # Save the generated image
    image.save('generatedimage.png')
    
    # Display the generated image on the user interface
    img = ImageTk.PhotoImage(image)
    img_placeholder.configure(image=img) 


trigger = ctk.CTkButton(height=40, width=120, text_font=("Arial", 15), text_color="black", fg_color="white", command=generate) 
trigger.configure(text="Generate")
trigger.place(x=206, y=60) 

app.mainloop()