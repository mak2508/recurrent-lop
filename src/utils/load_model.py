from src.config import Config
from src.nets import MLP, LSTM

def load_model(config: Config):
    # Initialize models based on config
    if config.model_type == "MLP":
        model = MLP(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate
        ).to(config.device)
    elif config.model_type == "LSTM":
        model = LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate
        ).to(config.device)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    return model