import torch
import torchvision.transforms as transforms
from PIL import Image
import timm

def load_model(model_path, device):
    # Load the pre-trained GC-ViT model
    model = timm.create_model('gcvit_tiny', pretrained=False, num_classes=2)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  
    return image

def predict(model, image, device):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()

        prob_Fake = torch.sigmoid(output[0].view(-1)).cpu().detach().numpy()[0]
        prob_Live = torch.sigmoid(output[0].view(-1)).cpu().detach().numpy()[1]
    return prob_Fake, prob_Live

def export_to_onnx(model, image, device, output_path):
    model.eval()  
    image = image.to(device)
    torch.onnx.export(model,  # Model to be exported
                      image,  # Sample input tensor
                      output_path,  # Path to save the ONNX file
                      export_params=True,  # Export model parameters
                    #   opset_version=11,  # ONNX opset version
                      do_constant_folding=True,  # Fold constant operations
                      input_names=['input'],  # Names of the input tensors
                      output_names=['output'],  # Names of the output tensors
                      dynamic_axes={'input': {0: 'batch_size'},  # Dynamic axes for input
                                    'output': {0: 'batch_size'}})  # Dynamic axes for output
    
def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = '/data/WORKSPACE_HS/TEST_CODE_huggingface/GCViT_For_S/Final_code_27_07/ckpt_/BEST_GCViT_TINY_orignal_image_01102024_old_code.pth'

    model = load_model(model_path, device)
    onnx_model_path = '/data/WORKSPACE_HS/TEST_CODE_huggingface/GCViT_For_S/Final_code_27_07/ckpt_/onnx/BEST_GCViT_TINY_orignal_image_01102024_old_code.onnx'
    sample_image_path = '/data/WORKSPACE_NISA/LightDarkNormal/DLN_linerm/4900_live_seg/ROI_DLN/dark/fingerprint_1_322.bmp'  # Replace this with your sample image path
    sample_image = preprocess_image(sample_image_path)

    export_to_onnx(model, sample_image, device, onnx_model_path)
    print("Model successfully exported to ONNX format.")

main()
