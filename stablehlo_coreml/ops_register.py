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
        raise ValueError(error_msg)

    if not issubclass(params[0].annotation, TranslationContext):
        raise ValueError(error_msg)

    # We identify the function by the type of operation it implements
    func._implements_hlo_op = params[1].annotation
    return func


def register_composite_op(composite_name: str):
    """Decorator that registers a method as the handler for a named StableHLO composite op.

    Usage::

        @register_composite_op("chlo.top_k")
        def _op_composite_chlo_top_k(self, context: TranslationContext, op: CompositeOp):
            ...
    """
    def decorator(func):
        func._implements_composite_op = composite_name
        return func
    return decorator


class StableHloOpsRegistry(type):
    def __init__(cls, name, bases, clsdict):
        super().__init__(name, bases, clsdict)

        cls._stablehlo_ops_registry = {}
        cls._composite_ops_registry = {}
        for _name, method in clsdict.items():
            op_type = getattr(method, '_implements_hlo_op', False)
            if callable(method) and op_type:
                if op_type in cls._stablehlo_ops_registry:
                    raise ValueError(f"StableHLO op {op_type} has been registered more than once!")
                cls._stablehlo_ops_registry[op_type] = method

            composite_name = getattr(method, '_implements_composite_op', None)
            if callable(method) and composite_name:
                if composite_name in cls._composite_ops_registry:
                    raise ValueError(f"Composite op '{composite_name}' has been registered more than once!")
                cls._composite_ops_registry[composite_name] = method

    def _dispatch_op(cls, self, context: TranslationContext, op):
        op_type = type(op)
        if op_type in self._stablehlo_ops_registry:
            return self._stablehlo_ops_registry[op_type](self, context, op)

        # Fall back: check if any registered type is a subclass of op_type.
        # This handles the case where the public API type (e.g. stablehlo.CompositeOp)
        # is a subclass of the internal type returned by the MLIR parser.
        for registered_type, method in self._stablehlo_ops_registry.items():
            if issubclass(registered_type, op_type):
                return method(self, context, op)

        raise ValueError(f"The StableHLO op {type(op)} has not been implemented!")

    def __call__(cls, *args, **kwargs):
        # Register the dispatch_op method
        instance = super().__call__(*args, **kwargs)
        instance.dispatch_op = cls._dispatch_op
        return instance
