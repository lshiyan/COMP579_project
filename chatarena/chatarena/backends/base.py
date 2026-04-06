from abc import abstractmethod
from typing import Dict, List, Optional, Type

from ..config import BackendConfig, Configurable
from ..message import Message


class IntelligenceBackend(Configurable):
    """An abstraction of the intelligence source of the agents."""

    stateful: Optional[bool] = None
    type_name: Optional[str] = None

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # registers the arguments with Configurable

    def __init_subclass__(cls, **kwargs):
        # check if the subclass has the required attributes
        for required in (
            "stateful",
            "type_name",
        ):
            if getattr(cls, required) is None:
                raise TypeError(
                    f"Can't instantiate abstract class {cls.__name__} without {required} attribute defined"
                )
        return super().__init_subclass__(**kwargs)

    def to_config(self) -> BackendConfig:
        self._config_dict["backend_type"] = self.type_name
        return BackendConfig(**self._config_dict)

    @abstractmethod
    def query(
        self,
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: Optional[str] = None,
        request_msg: Optional[Message] = None,
        *args,
        **kwargs,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def async_query(
        self,
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: Optional[str] = None,
        request_msg: Optional[Message] = None,
        *args,
        **kwargs,
    ) -> str:
        """Async querying."""
        raise NotImplementedError

    def get_message_embedding(self, message_text: str):
        raise NotImplementedError

    # reset the state of the backend
    def reset(self):
        if self.stateful:
            raise NotImplementedError
        else:
            pass


BACKEND_REGISTRY: Dict[str, Type[IntelligenceBackend]] = {}


def register_backend(cls: Type[IntelligenceBackend]) -> Type[IntelligenceBackend]:
    """Register a new backend."""
    assert cls.type_name is not None
    BACKEND_REGISTRY[cls.type_name] = cls
    return cls
