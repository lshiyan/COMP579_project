from chatarena.chatarena.arena import Arena
from chatarena.chatarena.config import ArenaConfig, BackendConfig

BACKEND_CONFIGS = {
    "openai-chat": {
        "backend_type": "openai-chat",
        "temperature": 0.9,
        "max_tokens": 100
    },
    "claude": {
        "backend_type": "claude",
        "temperature": 0.9,
        "max_tokens": 100
    },
    "gemini": {
        "backend_type": "gemini",
        "temperature": 0.9,
        "max_output_tokens": 100
    }
}

class ClosedSourceExperiment():
    
    def __init__(self, experiment_filepath: str, backend_name: str):
        
        self.experiment_filepath = experiment_filepath
        self.arena_config = ArenaConfig.load(experiment_filepath)

        for player in self.arena_config.players:
            player.backend = BackendConfig(BACKEND_CONFIGS[backend_name])
        
        self.arena = Arena.from_config(self.arena_config)