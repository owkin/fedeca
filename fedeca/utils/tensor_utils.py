"""Functions for tensors comparison."""
import numpy as np
import torch


def compare_tensors_lists(tensor_list_a, tensor_list_b, rtol=1e-5, atol=1e-8):
    """Compare list of tensors up to a certain precision.

    The criteria that is checked is the following: |x - y| <= |y| * rtol + atol.
    So there are two terms to consider. The first one is relative (rtol) and the second
    is absolute (atol). The default for atol is a bit low for float32 tensors. We keep
    the defaults everywhereto be safe except in the tests computed gradients wrt theory
    where we raise atol to 1e-6. It makes sens in this case because it matches the
    expected precision for slightly different float32 ops that should theoretically
    give the exact same result.

    Parameters
    ----------
    tensor_list_a: list
        a list of tensors
    tensor_list_b : list
        a list of tensors
    atol : float, optional
        absolute difference tolerance for tensor-to-tensor comparison. Default to 1e-5.
    rtol : float, optional
        relative difference tolerance for tensor-to-tensor comparison. Default to 1e-8.
    """
    if isinstance(tensor_list_a[0], torch.Tensor) and isinstance(
        tensor_list_b[0], torch.Tensor
    ):
        backend = "pytorch"

    elif isinstance(tensor_list_a[0], np.ndarray) and isinstance(
        tensor_list_b[0], np.ndarray
    ):
        backend = "numpy"

    else:
        raise RuntimeError(
            """Either the tensors you passed do not have the same type
               or the type is unsupported."""
        )

    if backend == "pytorch":
        assert all(
            [
                torch.allclose(u, d, rtol=rtol, atol=atol)
                for u, d in zip(tensor_list_a, tensor_list_b)
            ]
        )

    elif backend == "numpy":
        assert all(
            [
                np.allclose(u, d, rtol=rtol, atol=atol)
                for u, d in zip(tensor_list_a, tensor_list_b)
            ]
        )
