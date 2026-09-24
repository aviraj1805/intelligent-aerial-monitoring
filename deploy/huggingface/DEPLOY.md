# Deploying the demo to Hugging Face Spaces (free)

1. Create a new Space at https://huggingface.co/new-space (SDK: Gradio, hardware: CPU basic, free).
2. Clone it and copy these files from this repository into the Space folder:

   ```
   app.py
   iamars/                         (whole folder)
   requirements.txt
   deploy/huggingface/README.md    -> README.md   (Space config header)
   deploy/huggingface/packages.txt -> packages.txt
   ```

3. `git add . && git commit -m "IAMARS demo" && git push`.

On start-up the app downloads the model weights from this repository's GitHub
Release (`weights-v1`) and the example clip from Wikimedia Commons, so no large
files are committed to the Space.
