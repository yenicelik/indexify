from collections import defaultdict
import io
import tempfile

import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from IPython.core.display import HTML

import deepdoctection as dd

import torch
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList, VisionEncoderDecoderModel

from typing import Optional, List
import fitz
from pathlib import Path

from PIL import Image
from io import BytesIO
import requests


from ctransformers import AutoModelForCausalLM, AutoConfig
import time

from indexify_extractor_sdk import (
    Extractor,
    Feature,
    ExtractorSchema,
    Content,
)

# I could let the user decide which model to use. the 7B model will run very fast, we can go with this one for now (for debugging)
def get_airoboros_M_7B_312():
    # Got parameter recommendations from https://www.reddit.com/r/LocalLLaMA/comments/1343bgz/what_model_parameters_is_everyone_using/
    config = AutoConfig.from_pretrained('TheBloke/Airoboros-M-7B-3.1.2-GGUF')
    print('Config is', config)
    config.config.max_new_tokens = 512
    config.config.context_length = 4096 #96 # Could even be 8192 as I understand
    config.config.gpu_layers = 0 # TODO: This should be determined by the docker-image, or similar
    config.config.temperature = 0.7
    config.config.top_k = 50
    config.config.last_n_tokens = 256
    config.config.top_p = 0.1
    config.config.batch_size = 1
    # config.config.repetition_penalty = 0.85
    print(config)
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/Airoboros-M-7B-3.1.2-GGUF", model_file="airoboros-m-7b-3.1.2.Q8_0.gguf", model_type="mistral", config=config)
    return llm

def get_airoboros_L2_70B_312():
    # Got parameter recommendations from https://www.reddit.com/r/LocalLLaMA/comments/1343bgz/what_model_parameters_is_everyone_using/
    config = AutoConfig.from_pretrained('TheBloke/Airoboros-L2-70B-3.1.2-GGUF')
    print('Config is', config)
    config.config.max_new_tokens = 512
    config.config.context_length = 4096 #96 # Could even be 8192 as I understand
    config.config.gpu_layers = 9
    config.config.temperature = 0.7
    config.config.top_k = 50
    config.config.last_n_tokens = 256
    config.config.top_p = 0.1
    config.config.batch_size = 1
    # config.config.repetition_penalty = 0.85
    print(config)
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/Airoboros-L2-70B-3.1.2-GGUF", model_file="airoboros-l2-70b-3.1.2.Q4_K_M.gguf", model_type="mistral", config=config)
    print(llm)
    return llm

class RunningVarTorch:
    """
        Not 100% sure how Nougat works, but I suppose it runs a convergence algorithm for sampling.
        Will check it later
    """
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)


class StoppingCriteriaScores(StoppingCriteria):
    """
        Not 100% sure how Nougat works, but I suppose it runs a convergence algorithm for sampling.
        Will check it later
    """
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0

def rasterize_pdf(
    pdf: bytes,
    dpi: int = 96,
    pages=None,
) -> Optional[List[io.BytesIO]]:
    """
    Rasterize a PDF file to PNG images.
    This is a necessary step for the model to be compatible with Nougat,
    as it seems to be trained on this

    Args:
        pdf (Path): The path to the PDF file.
        dpi (int, optional): The output DPI. Defaults to 96.
        return_pil (bool, optional): Whether to return the PIL images instead of writing them to disk. Defaults to False.
        pages (Optional[List[int]], optional): The pages to rasterize. If None, all pages will be rasterized. Defaults to None.

    Returns:
        Optional[List[io.BytesIO]]: The PIL images if `return_pil` is True, otherwise None.
    """
    pillow_images = []
    try:
        if isinstance(pdf, bytes):
            pdf = fitz.open(stream=pdf)  # TODO: I suppose this should work
        if pages is None:
            pages = range(len(pdf))
        for i in pages:
            page_bytes: bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")
            pillow_images.append(io.BytesIO(page_bytes))
    except Exception as e:
        print("Exception: ", e)
        pass
    return pillow_images

class AdvancedInvoiceParsersExtractor(BaseModel):
    # No input except the file itself
    ...

class AdvancedInvoiceParsersExtractor(BaseEmbeddingExtractor):

    def __init__(self, max_context_length: int = 512, language = 'en'):
        super(PdfRawTextExtractor, self).__init__(max_context_length=512)
        # TODO: Must also make sure that we have >50GB of VRAM!
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load deepdoctection
        self.deepdoctetion_analyzer = dd.get_dd_analyzer()

        # Load Nougat
        self.nougat_processor = AutoProcessor.from_pretrained("facebook/nougat-base")
        self.nougat_model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
        self.nougat_model.to(self.device)

        # Load Llama70b (quantized), gpu if possible
        self.llm_model = get_airoboros_M_7B_312()

        # TODO: I should move the json type into the request of this extractor, that would be cool, then we can allow any unstructured data to be extracted (in theory). 
        # Gotta check lmql or similar to produce valid json each time (even if not semantically correct)
        # Finally, load the prompt
        self.prompt_prefix = """
            <|system|>
            Your goal is to extract structured information from the user's input 
            that matches the form described below. When extracting information 
            please make sure it matches the type information exactly. 
            Do not add any attributes that do not appear in the schema shown below.

            {
                "DocumentDescription": "string",
                "FromName": "string",
                "FromAddress": "string",
                "ToName": "string",
                "ToAddress": "string",
                "ToPhone": "string",
                "AmountDue": "number",
                "Currency": "string", // "default": "USD"
                "DueDate": "date",
                "PaymentDate": "date",
                "OrderNumber": "string",
                "VATAmount": "number",
                "InputTax": "number",
                "VATCode": "string",
                "Quarter": "string",
                "BookingDate": "date",
                "DocumentDate": "date",
                "DocumentNumber": "string",
                "Remarks": "string",
                "ConfidenceScore": {"type": "number", "minimum": 0, "maximum": 1}
            }

            Output MUST be a valid JSON object with the schema above.
            The following fields cannot be null: ["ToName", "ToAddress", "Description", "Amount", "Currency", "ConfidenceScore"]
            Do NOT add any clarifying information. 
            Do NOT add any fields that do not appear in the schema.
            Add a confidence score according to how confident you are with your JSON generation. 
            </s>\n<|user|>
        """

    def _get_deepdoctection_text_and_table(self, pdf_bytestring):
        # Currently just a blurb of what we need
        # TODO: Figure out how to read bytes; otherwise use tempfiles
        fp = tempfile.NamedTemporaryFile()
        fp.write(pdf_bytestring)
        fp.seek(0)
        df = self.analyzer.analyze(path=fp.name)

        df.reset_state()  # This method m
        # Deepdoctection
        doc = iter(df)
        page = next(doc)

        # Get the text
        dd_text = page.text
        print(dd_text)

        # Also extract these into arrays for metadata
        dd_tables = page.tables
        print(dd_tables)
        fp.close()

        return dd_text, dd_tables.csv

    def _predict(self, pdf_bytestring):
        dd_text, dd_tables = self._get_deepdoctection_text_and_table(pdf_bytestring)

        images = rasterize_paper(pdf=pdf_bytestring, return_pil=True)
        print(images)

        # TODO: Again, support multi-page documents!
        image = Image.open(images[0])

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        # autoregressively generate tokens, with custom stopping criteria (as defined by the Nougat authors)
        self.outputs = self.model.generate(
            pixel_values.to(self.device),
            min_length=1,
            max_length=3584,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
        )

        nougat_generated = self.processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
        print(nougat_generated)
        nougat_generated = self.processor.post_process_generation(nougat_generated, fix_markdown=False)
        print(nougat_generated)

        # Generate the prompt
        prompt = self.prompt_prefix + dd_text + "\n" + generated + """</s>\n<|assistant|>\n"""

        start = time.time()
        extracted_json = llm(prompt)
        print(extracted_json)
        print(f"Took {time.time() - start} seconds")
        return dd_text, nougat_generated, extracted_json

    def extract(
        self, content: List[Content], params: SimpleInvoiceParserInputParams
    ) -> List[List[Content]]:
        content_filebytes = [c.data for c in content]

        # TODO: Right now it only looks at the first image! We should probably flatten it and do it for each page!
        images = [convert_from_bytes(x)[0].convert("RGB") for x in content_filebytes]

        out = []
        for i, x in enumerate(content):
            print("i, x are: ", i, x)
            data = self._process_document(images[i])[0]  # Key 1 includes the image, which we ignore in this case
            out.append(
                [Content.from_text(
                    text="",  # TODO: Diptanu, what do we do for PDFs? Do you want to save the raw bytes too, I feel like this is unnecessary? Also, I felt like these would be stored in a database _before_ processing, not after
                    feature=Feature.metadata(value=data, name="invoice_extractor"),
                )]
            )
        return out

    def schemas(self) -> ExtractorSchema:
        """
        Returns a list of options for indexing.
        """
        input_params = PdfToMarkdownExtractor()
        # TODO If it's metadata, how do we extract things
        # This extractor does not return any embedding, only a dictionary!
        return ExtractorSchema(
            embedding_schemas={},
            input_params=json.dumps(input_params.model_json_schema()),
        )

