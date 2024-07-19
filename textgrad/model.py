from typing import Union
from textgrad.variable import Variable
from textgrad.autograd import LLMCall
from textgrad.autograd.function import Module
from textgrad.engine import EngineLM, get_engine
from .config import SingletonBackwardEngine

class BlackboxLLM(Module):
    def __init__(self, engine: Union[EngineLM, str] = None, system_prompt: Union[Variable, str] = None):
        """
        Initialize the LLM module.

        :param engine: The language model engine to use.
        :type engine: EngineLM
        :param system_prompt: The system prompt variable, defaults to None.
        :type system_prompt: Variable, optional
        """
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        if isinstance(system_prompt, str):
            system_prompt = Variable(system_prompt, requires_grad=False, role_description="system prompt for the language model")
        self.system_prompt = system_prompt
        self.llm_call = LLMCall(self.engine, self.system_prompt)

    def parameters(self):
        """
        Get the parameters of the blackbox LLM.

        :return: A list of parameters.
        :rtype: list
        """
        params = []
        if self.system_prompt:
            params.append(self.system_prompt)
        return params

    def forward(self, x: Variable) -> Variable:
        """
        Perform an LLM call.

        :param x: The input variable.
        :type x: Variable
        :return: The output variable.
        :rtype: Variable
        """
        return self.llm_call(x)


from typing import Union, List, Dict
from textgrad.variable import Variable
from textgrad.autograd.function import Module
from textgrad.engine import EngineLM, get_engine
from .config import SingletonBackwardEngine
from textgrad.autograd import MultimodalLLMCall, OrderedFieldsMultimodalLLMCall

class MultimodalBlackboxLLM(Module):
    def __init__(self, engine: Union[EngineLM, str] = None, system_prompt: Union[Variable, str] = None, fields: List[str] = None):
        """
        Initialize the Multimodal LLM module.

        :param engine: The language model engine to use.
        :type engine: EngineLM
        :param system_prompt: The system prompt variable, defaults to None.
        :type system_prompt: Variable, optional
        :param fields: The expected fields for the multimodal inputs, defaults to None.
        :type fields: List[str], optional
        """
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        self.engine = engine
        if isinstance(system_prompt, str):
            system_prompt = Variable(system_prompt, requires_grad=False, role_description="system prompt for the language model")
        self.system_prompt = system_prompt
        
        self.fields = fields
        if fields is None:
            self.llm_call = MultimodalLLMCall(self.engine, self.system_prompt)
        else:
            self.llm_call = OrderedFieldsMultimodalLLMCall(self.engine, fields, self.system_prompt)

    def parameters(self):
        """
        Get the parameters of the blackbox LLM.

        :return: A list of parameters.
        :rtype: list
        """
        params = []
        if self.system_prompt:
            params.append(self.system_prompt)
        return params

    def forward(self, inputs: Union[Variable, Dict[str, Variable]]) -> Variable:
        """
        Perform a multimodal LLM call.

        :param inputs: The input variable(s).
        :type inputs: Union[Variable, Dict[str, Variable]]
        :return: The output variable.
        :rtype: Variable
        """
        if isinstance(inputs, Variable):
            return self.llm_call([inputs])
        elif isinstance(inputs, dict):
            return self.llm_call(inputs)
        else:
            raise ValueError("Inputs should be either a Variable or a dictionary of Variables.")

