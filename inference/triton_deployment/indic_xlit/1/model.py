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
        
        # English to Indic Transliterator
        print("Loading English to Indic model...")
        en_indic_lang_pairs = "en-as,en-bn,en-brx,en-gom,en-gu,en-hi,en-kn,en-ks,en-mai,en-ml,en-mni,en-mr,en-ne,en-or,en-pa,en-sa,en-sd,en-si,en-ta,en-te,en-ur"
        self.en_indic_trans = Transliterator(
            data_bin_dir="/models/IndicXlit/corpus-bin",
            model_checkpoint_path="/models/IndicXlit/transformer-en-indic/indicxlit.pt",
            beam=beam,
            nbest=nbest,
            lang_pairs=en_indic_lang_pairs,
            batch_size=32 
        )

        # Indic to English Transliterator
        print("Loading Indic to English model...")
        indic_en_lang_pairs = "as-en,bn-en,brx-en,gom-en,gu-en,hi-en,kn-en,ks-en,mai-en,ml-en,mni-en,mr-en,ne-en,or-en,pa-en,sa-en,sd-en,si-en,ta-en,te-en,ur-en"
        self.indic_en_trans = Transliterator(
            data_bin_dir="/models/IndicXlit/corpus-bin",
            model_checkpoint_path="/models/IndicXlit/transformer-indic-en/indicxlit.pt",
            beam=beam,
            nbest=nbest,
            lang_pairs=indic_en_lang_pairs,
            batch_size=32 
        )
        
        # Rescoring enabled
        self.rescore_enabled = True
        self.word_prob_dict = {}
        
        if self.rescore_enabled:
            # Load dictionaries for supported languages including English for back-transliteration
            langs = ['en', 'as', 'bn', 'brx', 'gom', 'gu', 'hi', 'kn', 'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sd', 'si', 'ta', 'te', 'ur']
            base_path = "/models/IndicXlit/word_prob_dicts"
            print(f"Loading word_prob_dicts from {base_path}...")
            
            for lang in langs:
                try:
                    path = os.path.join(base_path, f"{lang}_word_prob_dict.json")
                    if os.path.exists(path):
                        with open(path, 'r', encoding='utf-8') as f:
                            self.word_prob_dict[lang] = json.load(f)
                    else:
                        print(f"Warning: Dict for {lang} not found at {path}")
                        self.word_prob_dict[lang] = {}
                except Exception as e:
                    print(f"Error loading dict for {lang}: {e}")
                    self.word_prob_dict[lang] = {}

        print("IndicXlit Model Initialized")

    def rescore(self, res_dict, result_dict, target_lang, alpha=0.9):
        """
        Rescoring logic ported from xlit_translit.py
        """
        if target_lang not in self.word_prob_dict:
            # specific dict not found, return empty or handle gracefully
            # effectively just returns best from model if dict is empty/missing
            return {}

        word_prob_dict = self.word_prob_dict[target_lang]
        if not word_prob_dict:
             return {}

        candidate_word_prob_norm_dict = {}
        candidate_word_result_norm_dict = {}

        input_data = {}
        for i in res_dict.keys():
            input_data[res_dict[i]['S']] = []
            for j in range(len(res_dict[i]['H'])):
                input_data[res_dict[i]['S']].append( res_dict[i]['H'][j][0] )
        
        output_data = {}

        for src_word in input_data.keys():
            candidates = input_data[src_word]
            candidates = [' '.join(word.split(' ')) for word in candidates]
            
            total_score = 0
            if src_word.lower() in result_dict.keys():
                for candidate_word in candidates:
                    if candidate_word in result_dict[src_word.lower()].keys():
                        total_score += result_dict[src_word.lower()][candidate_word]
            
            candidate_word_result_norm_dict[src_word.lower()] = {}
            for candidate_word in candidates:
                if total_score > 0 and src_word.lower() in result_dict and candidate_word in result_dict[src_word.lower()]:
                    candidate_word_result_norm_dict[src_word.lower()][candidate_word] = (result_dict[src_word.lower()][candidate_word]/total_score)
                else:
                    candidate_word_result_norm_dict[src_word.lower()][candidate_word] = 0

            candidates = [''.join(word.split(' ')) for word in candidates ]
            
            total_prob = 0 
            for candidate_word in candidates:
                if candidate_word in word_prob_dict.keys():
                    total_prob += word_prob_dict[candidate_word]        
            
            candidate_word_prob_norm_dict[src_word.lower()] = {}
            for candidate_word in candidates:
                if candidate_word in word_prob_dict.keys() and total_prob > 0:
                    candidate_word_prob_norm_dict[src_word.lower()][candidate_word] = (word_prob_dict[candidate_word]/total_prob)
                else:
                    candidate_word_prob_norm_dict[src_word.lower()][candidate_word] = 0
            
            temp_candidates_tuple_list = []
            candidates = input_data[src_word]
            candidates = [ ''.join(word.split(' ')) for word in candidates]
            
            for candidate_word in candidates:
                spaced_candidate = ' '.join(list(candidate_word))
                
                score_model = candidate_word_result_norm_dict[src_word.lower()].get(spaced_candidate, 0)
                score_dict = candidate_word_prob_norm_dict[src_word.lower()].get(candidate_word, 0)
                
                final_score = alpha * score_model + (1-alpha) * score_dict
                temp_candidates_tuple_list.append((candidate_word, final_score))

            temp_candidates_tuple_list.sort(key = lambda x: x[1], reverse = True )
            
            # For Triton we just want the top-1 best word
            if temp_candidates_tuple_list:
                output_data[src_word] = temp_candidates_tuple_list[0][0] # Return the joined word directly
            else:
                output_data[src_word] = candidates[0] if candidates else ""

        return output_data

    def execute(self, requests):
        """
        Receives a list of requests (a batch).
        """
        responses = []
        
        for request in requests:
            # Get inputs
            input_text = pb_utils.get_input_tensor_by_name(request, "TEXT")
            target_lang_tensor = pb_utils.get_input_tensor_by_name(request, "TARGET_LANG")
            source_lang_tensor = pb_utils.get_input_tensor_by_name(request, "SOURCE_LANG")
            
            # Triton strings are bytes, decode them
            texts = [t.decode('utf-8') for t in input_text.as_numpy().flatten()]
            target_lang = target_lang_tensor.as_numpy()[0][0].decode('utf-8').strip()
            # Default to 'en' if SOURCE_LANG is missing or empty for backward compatibility
            source_lang = "en"
            if source_lang_tensor is not None:
                source_lang = source_lang_tensor.as_numpy()[0][0].decode('utf-8').strip()
            
            # Determine which model to use and what the prefix should be
            if target_lang == "en":
                transliterator = self.indic_en_trans
                lang_token = source_lang
            else:
                transliterator = self.en_indic_trans
                lang_token = target_lang
            
            # Model is trained on WORDS. We must split sentences into words.
            all_words = []
            sentence_map = [] # store (num_words) for each sentence to reconstruct later
            
            for text in texts:
                # Split by space
                words = text.strip().split()
                if not words:
                    # Handle empty string case
                    sentence_map.append(0)
                else:
                    all_words.extend(words)
                    sentence_map.append(len(words))
            
            if not all_words:
                # If everything was empty
                output_texts = ["" for _ in texts]
            else:
                # We mimic xlit_translit.py's pre_process -> translate -> post_process

                # 1. Pre-processing
                processed_words = [word.lower() for word in all_words]
                processed_words = [' '.join(list(word)) for word in processed_words]
                processed_words = [f'__{lang_token}__ {word}' for word in processed_words]
                
                # 2. Translation (Fairseq)
                # Transliterator.translate takes a list of strings
                try:
                    # The custom_interactive.py 'translate' method returns a raw string formatted with S-id, H-id, etc.
                    result_str = transliterator.translate(processed_words)
                    
                    # 3. Post-processing (Parsing the result_str for N-best)
                    # We need to extract the 'H-' (Hypothesis) lines which contain the top prediction.
                    # Format: H-{id}\t{score}\t{hypothesis}
                    # id corresponds to the index in the batch.
                    lines = result_str.strip().split('\n')
                    
                    list_s = [line for line in lines if 'S-' in line]
                    list_h = [line for line in lines if 'H-' in line]
                    
                    list_s.sort(key = lambda x: int(x.split('\t')[0].split('-')[1]))
                    list_h.sort(key = lambda x: int(x.split('\t')[0].split('-')[1]))
                    
                    res_dict = {}
                    for s in list_s:
                        s_id = int(s.split('\t')[0].split('-')[1])
                        # The S- line contains the source word (preprocessed)
                        res_dict[s_id] = { 'S' : s.split('\t')[1] }
                        res_dict[s_id]['H'] = []
                        
                        for h in list_h:
                            h_id = int(h.split('\t')[0].split('-')[1])
                            if s_id == h_id:
                                # H line: H-0 \t score \t hypothesis
                                parts = h.split('\t')
                                hyp = parts[2]
                                score = float(parts[1])
                                res_dict[s_id]['H'].append((hyp, pow(2, score)))
                        
                        # Sort hypotheses by score desc
                        res_dict[s_id]['H'].sort(key=lambda x: float(x[1]), reverse=True)

                    # Build result_dict for rescoring
                    result_dict = {}
                    for i in res_dict.keys():
                        src = res_dict[i]['S']
                        result_dict[src] = {}
                        for j in range(len(res_dict[i]['H'])):
                             cand = res_dict[i]['H'][j][0]
                             prob = res_dict[i]['H'][j][1]
                             result_dict[src][cand] = prob
                    
                    # Apply Rescoring if enabled and dict exists
                    predictions = {}
                    
                    # Call rescore
                    rescored_map = self.rescore(res_dict, result_dict, target_lang)
                    
                    # Iterate to extract best result
                    for s_id in res_dict.keys():
                        src_word = res_dict[s_id]['S']
                        
                        if rescored_map and src_word in rescored_map:
                            # Rescoring returned a best candidate
                            final_cand = rescored_map[src_word]
                        else:
                            # Fallback to Top-1
                            final_cand = res_dict[s_id]['H'][0][0] if res_dict[s_id]['H'] else ""
                        
                        # Clean up spaces
                        val = final_cand.replace(' ', '')
                        # Clean up leading special characters/broken tokens
                        val = val.lstrip('.-_()[]{}|\\/!@#$%^&*+=,<>?;:"\'`~') 
                        predictions[s_id] = val

                    # Reconstruct sentences
                    output_texts = []
                    word_idx = 0
                    for count in sentence_map:
                        if count == 0:
                            output_texts.append("")
                        else:
                            # Join the translated words for this sentence
                            sentence_words = []
                            for _ in range(count):
                                val = predictions.get(word_idx, "")
                                sentence_words.append(val)
                                word_idx += 1
                            output_texts.append(" ".join(sentence_words))

                except Exception as e:
                    print(f"Error during inference: {e}")
                    output_texts = ["" for _ in texts]
                    err_tensor = pb_utils.Tensor("TRANSLITERATION", np.array(output_texts, dtype=object).reshape(-1, 1))
                    responses.append(pb_utils.InferenceResponse(output_tensors=[err_tensor], error=pb_utils.TritonError(str(e))))
                    continue

            # Pack response
            output_tensor = pb_utils.Tensor("TRANSLITERATION", np.array(output_texts, dtype=object).reshape(-1, 1))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self):
        """
        Clean up resources.
        """
        print("IndicXlit Model Finalized")

