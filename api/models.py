from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime


class ModelType(str, Enum):
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    BERT = "bert"
    GPT = "gpt"
    RESNET = "resnet"
    VGG = "vgg"
    UNKNOWN = "unknown"


class OptimizerType(str, Enum):
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    UNKNOWN = "unknown"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    UNKNOWN = "unknown"


class DatasetInfo(BaseModel):
    name: Optional[str] = None
    size: Optional[int] = None
    source: Optional[str] = None
    split_ratio: Optional[Dict[str, float]] = None  # {"train": 0.8, "val": 0.1, "test": 0.1}
    features: Optional[List[str]] = None


class ModelArchitecture(BaseModel):
    model_type: ModelType = ModelType.UNKNOWN
    layers: Optional[int] = None
    parameters: Optional[int] = None
    input_size: Optional[List[int]] = None
    output_size: Optional[List[int]] = None
    hidden_size: Optional[int] = None
    attention_heads: Optional[int] = None
    architecture_details: Optional[Dict[str, Any]] = None


class TrainingInfo(BaseModel):
    is_trained: bool = False
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    optimizer: Optional[OptimizerType] = OptimizerType.UNKNOWN
    loss_function: Optional[str] = None
    early_stopping: Optional[bool] = None
    regularization: Optional[Dict[str, Any]] = None  # {"dropout": 0.1, "l2": 0.01}


class PerformanceMetrics(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    loss: Optional[float] = None
    validation_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None


class FrameworkInfo(BaseModel):
    framework: Optional[str] = None  # "pytorch", "tensorflow", "keras", etc.
    version: Optional[str] = None
    gpu_used: Optional[bool] = None
    device: Optional[str] = None  # "cuda", "cpu", "mps"


class CodeStructure(BaseModel):
    language: Optional[str] = None
    main_functions: Optional[List[str]] = None
    classes: Optional[List[str]] = None
    imports: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None
    entry_points: Optional[List[str]] = None


class ResearchPaperInfo(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    methodology: Optional[str] = None
    contributions: Optional[List[str]] = None
    results: Optional[Dict[str, Any]] = None
    conclusions: Optional[str] = None


class ExtractedEntities(BaseModel):
    """Main structured data container"""
    # Model Information
    model_architecture: Optional[ModelArchitecture] = None
    training_info: Optional[TrainingInfo] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    framework_info: Optional[FrameworkInfo] = None
    
    # Dataset Information
    dataset_info: Optional[DatasetInfo] = None
    
    # Code Information
    code_structure: Optional[CodeStructure] = None
    
    # Research Paper Information
    research_info: Optional[ResearchPaperInfo] = None
    
    # General Information
    task_type: Optional[TaskType] = TaskType.UNKNOWN
    domain: Optional[str] = None  # "nlp", "computer_vision", "robotics", etc.
    file_type: Optional[str] = None  # "research_paper", "code_file", "notebook", etc.
    
    # Validation Information
    confidence_score: Optional[float] = None  # 0.0 to 1.0
    extraction_quality: Optional[str] = None  # "high", "medium", "low"
    missing_information: Optional[List[str]] = None
    potential_hallucinations: Optional[List[str]] = None
    
    # Raw extracted text for reference
    raw_extracted_text: Optional[str] = None
    
    # Metadata
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    file_path: Optional[str] = None
    extraction_method: Optional[str] = None


class ValidationResult(BaseModel):
    """Validation results for extracted entities"""
    is_valid: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    validation_details: Dict[str, Any]
    missing_entities: List[str]
    potential_errors: List[str]
    suggestions: List[str]


class ExtractionResponse(BaseModel):
    """Final response with extracted entities and validation"""
    entities: ExtractedEntities
    validation: ValidationResult
    success: bool
    message: str
