from renikud_onnx import Renikud

# Load the model
model = Renikud('nikud_model.onnx')

# Plain Hebrew text
text = "הוא רצה את זה גם, אבל היא רצה מהר והקדימה אותו!"

# Add nikud (diacritical marks)
nikud_text = model.add_diacritics(text)

print(f"Input:  {text}")
print(f"Output: {nikud_text}")