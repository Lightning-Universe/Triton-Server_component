# Serve an Image Classifier built with Torch Vision using Lightning Triton Serve

### Step 1

Install lightning using

```bash
pip install -U lightning
```

### Step 2

Clone this repo and cd to examples/torchvision

```bash
git clone https://github.com/Lightning-AI/LAI-Triton-Server-Component.git
cd LAI-Triton-Server-Component/examples/torchvision
```

### Step 3

run the component using below command (if you need to run it in cloud, add `--cloud` at the end of the command)

```bash
lightning run app serve_torchvision.py --setup
```

### Step 4

Once the server is up and running, you should see the client code in the Lightning App UI.
Run that in a terminal to interact with the server. We have already copied that code
into `client.py`, incase you need. Run it with

```bash
python client.py
```
