# Serve Stable Diffusion using Lightning Triton Serve

### Step 1

Install lightning using

```bash
pip install -U lightning
```

### Step 2

Clone this repo and cd to examples/stable-diffusion

```bash
git clone https://github.com/Lightning-AI/LAI-Triton-Server-Component.git
cd LAI-Triton-Server-Component/examples/stable-diffusion
```


### Step 3

run the component using below command (if you need to run it in clud, add `--cloud` at the end of the command)

```bash
lightning run app serve_stable_diffusion.py --setup
```

Note: This is going to download the stable diffusion weights from s3 and can take time depends on your internet speed.

### Step 4

Once the server is up and running, you should see the client code in the Lightning App UI.
Run that in a terminal to interact with the server. We have already copied that code 
into `client.py`, incase you need. Run it with

```bash
python client.py
```
