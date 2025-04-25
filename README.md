# CMT analiza mnożników pod wycene 🔍

A secure Streamlit app for comparing M&A transactions using OpenAI embeddings and web scraping.

## 🔐 Secrets Setup for Streamlit Cloud

Add this under Settings > Secrets:

```toml
[openai]
api_key = "sk-..."
```

## 🧪 Running Locally

To test locally, create a file at:
```
.streamlit/secrets.toml
```

With this content:
```toml
[openai]
api_key = "sk-..."  # Your real OpenAI key
```

This avoids the `StreamlitSecretNotFoundError` when not running in the cloud.
