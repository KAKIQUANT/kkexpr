from functools import wraps
import pandas as pd
from typing import Callable, Any, List, Union
from .expr_exceptions import InvalidInputError

def calc_by_date(func: Callable) -> Callable:
    """Decorator for calculating expressions by date.
    
    This decorator groups the input series by date and applies the function to each group.
    It handles both single and multiple series inputs.
    
    Args:
        func: The function to be wrapped
        
    Returns:
        Wrapped function that processes data by date
        
    Raises:
        InvalidInputError: If no series arguments are provided
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> pd.Series:
        other_args: List[Any] = []
        se_args: List[pd.Series] = []
        se_names: List[str] = []
        
        for arg in args:
            if not isinstance(arg, pd.Series):
                other_args.append(arg)
            else:
                se_args.append(arg)
                se_names.append(arg.name)
                
        if not se_args:
            raise InvalidInputError(f"No Series arguments provided to {func.__name__}")
            
        if len(se_args) == 1:
            ret = se_args[0].groupby(level=0, group_keys=False).apply(
                lambda x: func(x, *other_args, **kwargs)
            )
        else:
            df = pd.concat(se_args, axis=1)
            df.index = se_args[0].index
            ret = df.groupby(level=0, group_keys=False).apply(
                lambda sub_df: func(*[sub_df[name] for name in se_names], *other_args)
            )
            ret.index = df.index
            
        return ret
    return wrapper

def calc_by_symbol(func: Callable) -> Callable:
    """Decorator for calculating expressions by symbol.
    
    This decorator groups the input series by symbol and applies the function to each group.
    It handles both single and multiple series inputs, and maintains proper naming of results.
    
    Args:
        func: The function to be wrapped
        
    Returns:
        Wrapped function that processes data by symbol
        
    Raises:
        InvalidInputError: If no series arguments are provided
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Union[pd.Series, None]:
        other_args: List[Any] = []
        se_args: List[pd.Series] = []
        se_names: List[str] = []
        
        for i, arg in enumerate(args):
            if not isinstance(arg, pd.Series):
                other_args.append(arg)
            else:
                se_args.append(arg)
                if arg.name:
                    se_names.append(arg.name)
                else:
                    name = f'arg_{i}'
                    arg.name = name
                    se_names.append(name)
                    
        if not se_args:
            raise InvalidInputError(f"No Series arguments provided to {func.__name__}")
            
        if len(se_args) == 1:
            ret = se_args[0].groupby(level=1, group_keys=False).apply(
                lambda x: func(x, *other_args, **kwargs)
            )
            ret.name = f"{func.__name__}_{se_names[0]}"
            if other_args:
                ret.name += str(other_args[0])
        else:
            df = pd.concat(se_args, axis=1)
            df.index = se_args[0].index
            
            unique_level1 = df.index.get_level_values(1).unique()
            
            if len(unique_level1) == 1:
                ret = func(*[df[name] for name in se_names], *other_args)
            else:
                ret = df.groupby(level=1, group_keys=False).apply(
                    lambda sub_df: func(*[sub_df[name] for name in se_names], *other_args)
                )
                ret.index = df.index
                
        return ret
    return wrapper
