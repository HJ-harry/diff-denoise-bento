import numpy as np
import bentoml
from typing import TYPE_CHECKING
from bentoml.io import NumpyNdarray, Image

if TYPE_CHECKING:
    from PIL.Image import Image

task_name = "denoising"

denoise_runner = bentoml.models.get(f"{task_name}:latest").to_runner()

svc = bentoml.Service(
    name=task_name, runners=[denoise_runner]
)

@svc.api(input=Image(), output=Image())
async def denoise(img: Image) -> Image:
    assert isinstance(img, Image)
    # 0. PIL -> ndarray -> torch tensor
    img = np.array(img)
    img = torch.from_numpy(img)
    h, w = img.shape
    img = img.view(1, 1, h, w)
    
    # 1. NFE (diffusion)
    denoised_img = await denoise_runner.async_run(img)
    
    # 2. post-processing save back to PIL
    denoised_img = denoised_img.squeeze().numpy()
    denoised_img = Image.fromarray(np.unit8(denoised_img * 255.))
    return denoised_img
