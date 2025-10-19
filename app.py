import os

# Import Gradio Blocks app and startup loader from web package
from web.app import app as gradio_app  # Blocks instance
from web.app import _startup_load


# Ensure model and classes are loaded at import time for Spaces
try:
	model, classes = _startup_load()
except Exception:
	# Allow UI to load even if model isn't available yet; user can switch/load later
	pass


# Expose the Gradio app object for Hugging Face Spaces
app = gradio_app


if __name__ == "__main__":
	app.launch()


