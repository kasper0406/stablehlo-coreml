import inspect

from .translation_context import TranslationContext


def register_stablehlo_op(func):
    # Check the signature
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Exclude 'self' from the parameters
    params = params[1:]

    error_msg = "HLO op implementations should take parameters of exactly " \
                "(context: TranscriptionContext, op: <HLO_OP_TYPE>)"
    if len(params) != 2:
        raise TypeError(error_msg)

    if not issubclass(params[0].annotation, TranslationContext):
        raise TypeError(error_msg)

    # We identify the function by the type of operation it implements
    func._implements_hlo_op = params[1].annotation
    return func


class StableHloOpsRegistry(type):
    def __init__(cls, name, bases, clsdict):
        super().__init__(name, bases, clsdict)

        cls._stablehlo_ops_registry = {}
        for name, method in clsdict.items():
            op_type = getattr(method, '_implements_hlo_op', False)
            if callable(method) and op_type:
                if op_type in cls._stablehlo_ops_registry:
                    raise TypeError(f"StableHLO op {op_type} has been registered more than once!")
                cls._stablehlo_ops_registry[op_type] = method

    def _dispatch_op(cls, self, context: TranslationContext, op):
        if type(op) not in self._stablehlo_ops_registry:
            raise TypeError(f"The StableHLO op {type(op)} has not been implemented!")

        op_method = self._stablehlo_ops_registry[type(op)]
        return op_method(self, context, op)

    def __call__(cls, *args, **kwargs):
        # Register the dispatch_op method
        instance = super().__call__(*args, **kwargs)
        setattr(instance, 'dispatch_op', cls._dispatch_op)
        return instance
