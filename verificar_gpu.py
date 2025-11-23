import torch

print(f"¿PyTorch ve la GPU? -> {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Nombre de tu bestia: {torch.cuda.get_device_name(0)}")
    print("¡Todo listo para entrenar rápido!")
else:
    print("Sigue usando CPU. Revisa la instalación del Paso 2.")