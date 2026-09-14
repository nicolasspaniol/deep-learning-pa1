# AI usage disclosure

LLMs (Claude, ChatGPT, OpenAI Codex, Gemini, all to some extent) where used mainly for these tasks, ordered by frequency:
- Debugging exceptions and unexpected model outputs
- Creation of custom pytorch `Dataset`s adapted for some questions (eg with tiling, center offsets, corruption)
- Rewriting Jupyter notebook contents as standard Python files
- Writing unit tests
- Clarifying understanding of some of the project subjects and interpretation of the instructions
- Learning about data science project organization and good practices

For reference, these are all the titles of our relevant Claude chats at the end of the project:
- BBBC038 test dataset labeling issue
- Merging overlapping instances across tiles
- Creating noisy BBBC038 dataset wrapper
- TiledBBBC038 with random_split sequence preservation
- Adapting model training for foreground mask prediction
- Project tree structure
- Custom torch dataset with image slicing
- Code performance bottleneck analysis
- Converting mAP evaluation code to module function
- Decoupling plotting functions in Python modules
- Rewriting instance IoU matrix without numpy
- Creating coordinate tensors in PyTorch
- Modelo com erro crescente nos epochs
- Adapting ResUNet for C path training
- PyTorch DataLoader GPU data transfer
- Downloading datasets to class folders
- Simplifying IoU and Dice validation code
- Executing functions on dataset coordinates
- Unit tests for mask scoring functions
- Debugging Colab memory usage
- Batch norm running mean dimension mismatch
- Carregando dados no Jupyter Notebook
- Model training loss stuck at zero
- CrossEntropyLoss input value range
- Validação de implementação ResUNet
- ResUNet upsampling in PyTorch
- Synthetic ellipse dataset with instance segmentation
- Multiple inputs in PyTorch forward method

And for ChatGPT
- Explicación del Max-Unpooling y su Cuadrícula
- Entendiendo el Stride en ResUNet
- Guía para evaluar pérdida focal

For the actual prompts, see these as examples:
- [Adapting model training for foreground mask prediction](https://claude.ai/share/978804aa-f641-4b1f-9bea-cbb755189747)
- [Converting mAP evaluation code to module function](https://claude.ai/share/01706bb0-a2b5-4922-8bc5-60545d50d7de)
- [TiledBBBC038 with random_split sequence preservation](https://claude.ai/share/1cf125a8-eedc-427c-8232-ef3c86cc77f9)
