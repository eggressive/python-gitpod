# Gitpod Launchpad for Prompt Engineering with Llama 2

## Standard setup for gitpod with Python 3.12 and llama 2

### Chat Models by Together.ai API

[Click here](https://docs.together.ai/docs/inference-models) to access the Togeher.ai inference models documentation.

### Env vars

Set `gp env TOGETHER_API_KEY='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'`

`gp env` can only interact with the persistent environment variables for this repository, not the environment variables of your terminal. If you want to set that environment variable in your terminal, you can do so using `-e`:

```bash
eval $(gp env -e foo=bar)
```
