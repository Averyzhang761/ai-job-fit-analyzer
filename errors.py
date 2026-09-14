class JobFitError(RuntimeError):
    error_type = "job_fit_error"


class ModelRequestError(JobFitError):
    error_type = "request_error"


class ModelResponseError(JobFitError):
    error_type = "response_error"


class ModelOutputTruncatedError(ModelResponseError):
    error_type = "output_truncated"


class ModelContractError(JobFitError):
    error_type = "contract_error"
