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

For running the component locally, use below command

```bash
lightning run app serve_stable_diffusion.py --setup
```
Note: This is going to download the stable diffusion weights from s3 and can take time depends on your internet speed.


For running it in cloud, use below command

```bash
lightning run app serve_stable_diffusion.py --setup --cloud
```


### Step 4

Once the server is running, Lightning will open a new tab in your browser. To access your server you can use
the code snippet shown in that tab or the pre-packaged client.py

```bash
python client.py
```
