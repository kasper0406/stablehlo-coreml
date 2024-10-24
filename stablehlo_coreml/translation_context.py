from coremltools.converters.mil import mil


class TranslationContext:
    def __init__(self):
        self._path = []
        self.seen_paths = set()
        self.variables = {}  # Nested map: path -> variable -> mil var

    def push_function(self, name: str):
        counter = 0
        ctx_name = name
        while True:
            new_path = self._path + [ctx_name]
            if "/".join(new_path) in self.seen_paths:
                # Ensure that the new context name is in fact unique
                # A collision can happen if the same function is called twice
                ctx_name = f"{name}_{counter}"
                counter += 1
            else:
                self._path.append(ctx_name)
                self.seen_paths.add(self.path())
                return ctx_name

    def pop_function(self):
        self.variables.pop(self.path())
        self._path.pop()

    def add_variable(self, name: str, mil_var: mil.Var):
        path = self.path()
        if path not in self.variables:
            self.variables[path] = {}

        if name in self.variables[path]:
            raise ValueError(f"Variable {name} is already defined in path {path}")
        self.variables[path][name] = mil_var

    def add_result(self, hlo_result, result: mil.Var):
        result_name = hlo_result.get_name()
        self.add_variable(result_name, result)

        def validate_shapes(hlo_shape: tuple, mil_shape: tuple):
            if hlo_shape == tuple() and (mil_shape == tuple() or mil_shape == (1, )):
                return True
            if hlo_shape == mil_shape:
                return True

            raise ValueError(f"The HLO result shape `{hlo_shape}` is different from the actual MIL result shape `{mil_shape}`")

        hlo_shape = tuple(hlo_result.type.shape)
        mil_shape = tuple(result.shape)
        validate_shapes(hlo_shape=hlo_shape, mil_shape=mil_shape)

    def __getitem__(self, name: str):
        # Walk up along the path list to find the first correctly named variable in scope
        path = self._path.copy()
        while True:
            ctx = self.variables["/".join(path)]
            if name in ctx:
                return ctx[name]
            if len(path) == 0:
                raise ValueError(f"Variable with name {name} is not defined in path {path}")
            path.pop()

    def path(self) -> str:
        return "/".join(self._path)
