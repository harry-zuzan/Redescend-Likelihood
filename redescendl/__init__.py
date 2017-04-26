from redescend_likelihood \
	import redescend_residuals_normal_1d as redescend_normal1

from redescend_likelihood \
	import redescend_residuals_normal_2d as redescend_normal2

from redescend_likelihood \
	import redescend_residuals_normal_3d as redescend_normal3

from redescend_likelihood \
	import get_redescend_weights_normal_1d as get_weights_normal1

from redescend_likelihood \
	import get_redescend_weights_normal_2d as get_weights_normal2

from redescend_likelihood \
	import get_redescend_weights_normal_3d as get_weights_normal3



from redescend_likelihood \
	import redescend_residuals_logistic_2d as redescend_logistic2

from redescend_likelihood \
	import get_redescend_weights_logistic_2d as get_weights_logistic2

__all__ = ["redescend_normal2", "redescend_normal1"]
__all__ += ["get_weights_normal2", "get_weights_normal1"]
__all__ += ["redescend_logistic2"]

del redescend_likelihood
