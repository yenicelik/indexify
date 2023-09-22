# ! apt install tesseract-ocr
# ! apt install libtesseract-dev

# ! pip install transformers
# ! pip install torch torchvision
# ! pip install Pillow
# ! pip install pytesseract
# import pytesseract

from transformers import LayoutLMv2ForQuestionAnswering, LayoutLMv2Tokenizer
from PIL import Image
import torch
import json 
from typing import List, Any, Union
from .base_extractor import Extractor, ExtractorInfo, Content, ExtractedAttributes, ExtractedEmbedding
from pydantic import BaseModel
from transformers import pipeline

class QuestionAnsweringInputParams(BaseModel):
    question: str

class InvoiceQA(Extractor):
    def __init__(self):
        # Keep it stupid simple, only use a standard pipeline for now, instead of fancy models
        # TODO: We can probably run multiple models, to get uncertainty estimates. Expensive, but could be worth it. 
        self.model = pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa",
        )
        self.top_k = 5
        
    def answer_question(self, question: str, image_path: str):
        return self.model(
            image_path,
            question,
            # maximum number of characters of the question you can pose, the shorter the better
            max_question_len=128,
            top_k=self.top_k
        )

    def extract(self, content: List[Content], params: dict[str, Any]) -> List[Union[ExtractedEmbedding, ExtractedAttributes]]:
        extracted_data = []
        for c in content:
            answer = self.answer_question(params['question'], c.data)
            extracted_data.append(ExtractedAttributes(content_id=c.id, json=json.dumps({'answer': answer})))
        return extracted_data

    def info(self) -> ExtractorInfo:
        # TODO: How do i denote a list; are number-type names correct?
        schema = {"answer": "string", "score": "float", "start": "integer", "end": "integer"}
        schema_json = json.dumps(schema)
        return ExtractorInfo(
            name="InvoiceQA",
            description="Question Answering for Invoices",
            output_datatype="attributes",
            input_params=json.dumps(QuestionAnsweringInputParams.schema_json()),
            output_schema=schema_json,
        )