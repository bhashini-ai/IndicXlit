
import numpy as np
import sys
import os
import json
import triton_python_backend_utils as pb_utils

# Add the model directory to sys.path so we can import dependencies
# Assuming the file structure:
# triton_deployment/indic_xlit/1/model.py
# triton_deployment/indic_xlit/1/custom_interactive.py
# triton_deployment/indic_xlit/1/xlit_translit.py (optional, but we might inline the logic or use custom_interactive directly)

# We need to make sure 'custom_interactive' and other local modules can be imported
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

from custom_interactive import Transliterator

class TritonPythonModel:
    def initialize(self, args):
        """
        Called when the model is being loaded.
        """
        self.model_config = json.loads(args['model_config'])
        
        # Load the model
        # The paths below are relative to 'model.py' location in the repository structure
        # We assume the user has copied 'corpus-bin' and 'transformer' folders to this directory
        
        # NOTE: Model loading args from original xlit_translit.py
        # beam=5, nbest=5 is standard for good quality
        beam = 5
        nbest = 5
        
        self.transliterator = Transliterator(
            data_bin_dir="/models/IndicXlit/corpus-bin",
            model_checkpoint_path="/models/IndicXlit/transformer/indicxlit.pt",
            beam=beam,
            nbest=nbest,
            batch_size=32 
        )
        
        # Rescoring is currently DISABLED for performance and due to missing dependencies in the source repo
        # If enabled, we would load word_prob_dicts here.
        self.rescore = False

        print("IndicXlit Model Initialized")

    def execute(self, requests):
        """
        Receives a list of requests (a batch).
        """
        responses = []
        
        for request in requests:
            # Get inputs
            input_text = pb_utils.get_input_tensor_by_name(request, "TEXT")
            target_lang_tensor = pb_utils.get_input_tensor_by_name(request, "TARGET_LANG")
            
            # Triton strings are bytes, decode them
            texts = [t.decode('utf-8') for t in input_text.as_numpy().flatten()]
            target_lang = target_lang_tensor.as_numpy()[0][0].decode('utf-8')
            
            # Inference
            # We assume 'texts' is a list of words or sentences. 
            # The original logic handles splitting sentences into words.
            # We will use the transliterator's logic but adapted.
            
            # Preprocess is built-in to 'translate' in our custom usage below or we mimic xlit_translit
            # xlit_translit.py does: pre_process -> translate -> post_process
            
            # 1. Pre-processing (mimicking xlit_translit.py logic)
            # small caps
            processed_words = [word.lower() for word in texts]
            # space separated chars
            processed_words = [' '.join(list(word)) for word in processed_words]
            # add language token
            processed_words = [f'__{target_lang}__ {word}' for word in processed_words]
            
            # 2. Translation (Fairseq)
            # Transliterator.translate takes a list of strings
            try:
                # The custom_interactive.py 'translate' method returns a raw string formatted with S-id, H-id, etc.
                # We need to parse this or modify custom_interactive.py.
                # To minimize code changes to the original repo files, we will parse the output string.
                # However, calling 'translate' directly prints to stdout which is bad for Triton.
                # We should look at 'custom_interactive.py' again.
                # It calls 'self.task.inference_step'. 
                
                # Let's try to capture the output or use the internal methods if possible.
                # 'translate' returns 'result_str'.
                
                result_str = self.transliterator.translate(processed_words)
                
                # 3. Post-processing (Parsing the result_str)
                # We need to extract the 'H-' (Hypothesis) lines which contain the top prediction.
                # Format: H-{id}\t{score}\t{hypothesis}
                # id corresponds to the index in the batch.
                
                lines = result_str.strip().split('\n')
                predictions = {}
                
                for line in lines:
                    if line.startswith('H-'):
                        parts = line.split('\t')
                        idx = int(parts[0].split('-')[1])
                        score = float(parts[1])
                        transliteration = parts[2]
                        
                        # We only want the best score for each index (since nbest might be > 1)
                        # The output is usually sorted but let's be safe.
                        if idx not in predictions:
                            predictions[idx] = transliteration
                        # Attempt to join spaces (reverse of space separation)
                        # The model outputs space-separated characters usually? 
                        # Looking at xlit_translit.post_process:
                        # transliterated_word_list = [''.join(word.split(' ')) for word in transliterated_word_list]
                
                # Reconstruct output list in order
                output_texts = []
                for i in range(len(texts)):
                    val = predictions.get(i, "")
                    # Remove spaces between characters
                    val = val.replace(' ', '')
                    output_texts.append(val)
                
                # Pack response
                output_tensor = pb_utils.Tensor("TRANSLITERATION", np.array(output_texts, dtype=object).reshape(-1, 1))
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
                
            except Exception as e:
                # Error handling
                print(f"Error during inference: {e}")
                # Return empty or error - for now return empty strings to avoid crashing everything
                err_tensor = pb_utils.Tensor("TRANSLITERATION", np.array(["" for _ in texts], dtype=object).reshape(-1, 1))
                responses.append(pb_utils.InferenceResponse(output_tensors=[err_tensor], error=pb_utils.TritonError(str(e))))

        return responses

    def finalize(self):
        """
        Clean up resources.
        """
        print("IndicXlit Model Finalized")

