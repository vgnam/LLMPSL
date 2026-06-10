from __future__ import annotations

import os
import sys
import pytz
import json
import logging
import hashlib
from threading import Lock
from datetime import datetime

from ...base import Function


class ProfilerBase:
    _num_samples = 0
    _sensitive_parameter_names = (
        'api_key',
        'apikey',
        'credential',
        'password',
        'secret',
        'token',
    )

    process_start_time = datetime.now(pytz.timezone("Asia/Bangkok"))
    result_folder = process_start_time.strftime("%Y%m%d_%H%M%S")

    def __init__(self,
                 log_dir: str | None = None,
                 evaluation_name="Problem",
                 method_name="Method",
                 initial_num_samples=0,
                 log_style='complex',
                 **kwargs):

        assert log_style in ['simple', 'complex']
        self.__class__._num_samples = initial_num_samples
        self._log_dir = log_dir
        self._log_style = log_style
        self._cur_best_function = None
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = float('-inf')
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0

        self._evaluation_name = evaluation_name
        self._method_name = method_name
        self._parameters = None
        self._logger_txt = logging.getLogger('root')
        self._log_dir = os.path.join(log_dir,
                                     self.__class__.result_folder + '_' +
                                     self._evaluation_name + '_' +
                                     self._method_name)
        self._log_dir = kwargs.get('final_log_dir', self._log_dir)

        # lock for multi-thread invoking self.register_function(...)
        self._register_function_lock = Lock()

    def record_parameters(self, llm, prob, method):
        self._parameters = [llm, prob, method]
        self._create_log_path()

    def register_function(self, function: Function, program='', *, resume_mode=False):
        """Record an obtained function. This is a synchronized function.
        """
        try:
            self._register_function_lock.acquire()
            self.__class__._num_samples += 1
            self._record_and_verbose(function, resume_mode=resume_mode)
            self._write_json(function, program=program)
        finally:
            self._register_function_lock.release()

    def finish(self):
        pass

    def get_logger(self):
        return self._logger_txt

    def resume(self, *args, **kwargs):
        pass

    def _write_json(self, function: Function, program='', *, record_type='history', record_sep=200):
        """
            Write function data to a JSON file.

            Parameters:
                function (Function): The function object containing score and string representation.
                record_type (str, optional): Type of record, 'history' or 'best'. Defaults to 'history'.
                record_sep (int, optional): Separator for history records. Defaults to 200.
            """
        assert record_type in ['history', 'best']

        if not self._log_dir:
            return

        sample_order = getattr(self.__class__, '_num_samples', 0)
        content = {
            'sample_order': sample_order,
            'function': str(function),
            'score': function.score,
            'program': program,
        }
        if hasattr(function, 'island_id'):
            content['island_id'] = function.island_id

        if record_type == 'history':
            lower_bound = (sample_order // record_sep) * record_sep
            upper_bound = lower_bound + record_sep
            filename = f'samples_{lower_bound}~{upper_bound}.json'
        else:
            filename = 'samples_best.json'

        path = os.path.join(self._samples_json_dir, filename)

        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(content)

        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def _record_and_verbose(self, function, *, resume_mode=False):
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score

        if not resume_mode:
            # log attributes of the function
            if self._log_style == 'complex':
                print(f'================= Evaluated Function =================')
                print(f'{self._function_summary(function)}')
                print(f'Score        : {str(score)}')
                print(f'Sample time  : {str(sample_time)}')
                print(f'Evaluate time: {str(evaluate_time)}')
                print(f'Sample orders: {str(self.__class__._num_samples)}')
                print(f'======================================================\n')
            else:
                if score is None:
                    print(f'Sample{self.__class__._num_samples}: Score=None    Cur_Best_Score={self._cur_best_program_score: .3f}')
                else:
                    print(f'Sample{self.__class__._num_samples}: Score={score: .3f}     Cur_Best_Score={self._cur_best_program_score: .3f}')

        # update statistics about function
        if score is not None:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time is not None:
            self._tot_sample_time += sample_time

        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

    def _function_summary(self, function: Function) -> str:
        function_str = str(function).strip('\n')
        function_hash = hashlib.sha1(function_str.encode('utf-8')).hexdigest()[:10]
        body = getattr(function, 'body', '') or ''
        body_lines = len([line for line in body.splitlines() if line.strip()])
        signature = f"def {getattr(function, 'name', '<unknown>')}({getattr(function, 'args', '')})"
        return (
            f'Function     : {signature}\n'
            f'Body lines   : {body_lines}\n'
            f'Code chars   : {len(function_str)}\n'
            f'Code sha1    : {function_hash}\n'
            f'Algorithm    : {self._short_text(getattr(function, "algorithm", None), 240)}\n'
            f'------------------------------------------------------'
        )

    @staticmethod
    def _short_text(value, limit=240):
        if value is None:
            return ''
        text = ' '.join(str(value).split())
        if len(text) <= limit:
            return text
        return text[:limit - 3] + '...'

    @classmethod
    def _safe_parameter_value(cls, attr, value):
        attr_lower = str(attr).lower()
        if any(name in attr_lower for name in cls._sensitive_parameter_names):
            return '<redacted>'
        if isinstance(value, dict):
            return {
                key: cls._safe_parameter_value(key, item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            sanitized = [cls._safe_parameter_value(attr, item) for item in value]
            return tuple(sanitized) if isinstance(value, tuple) else sanitized
        return value

    def _create_log_path(self):
        self._samples_json_dir = os.path.join(self._log_dir, 'samples')
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._samples_json_dir, exist_ok=True)

        file_name = self._log_dir + '/run_log.txt'
        file_mode = 'a' if os.path.isfile(file_name) else 'w'

        self._logger_txt.setLevel(level=logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

        for hdlr in self._logger_txt.handlers[:]:
            self._logger_txt.removeHandler(hdlr)

        # add handler
        fileout = logging.FileHandler(file_name, mode=file_mode)
        fileout.setLevel(logging.INFO)
        fileout.setFormatter(formatter)
        self._logger_txt.addHandler(fileout)
        self._logger_txt.addHandler(logging.StreamHandler(sys.stdout))

        # write initial parameters
        llm = self._parameters[0]
        prob = self._parameters[1]
        method = self._parameters[2]

        self._logger_txt.info("==================================LLM Parameters===============================")
        self._logger_txt.info(f"LLM: {llm.__class__.__name__}")
        for attr, value in llm.__dict__.items():
            if attr not in ['_functions']:
                self._logger_txt.info(f"{attr}: {self._safe_parameter_value(attr, value)}")

        self._logger_txt.info("==================================Problem Parameters===============================")

        self._logger_txt.info(f"Problem: {prob.__class__.__name__}")
        for attr, value in prob.__dict__.items():
            if attr not in ['template_program', '_datasets']:
                self._logger_txt.info(f"{attr}: {self._safe_parameter_value(attr, value)}")

        self._logger_txt.info("==================================Method Parameters===============================")

        self._logger_txt.info(f"Method: {method.__class__.__name__}")
        for attr, value in method.__dict__.items():
            if attr not in ['llm', '_evaluator', '_profiler', '_template_program_str', '_template_program', '_function_to_evolve', '_population', '_sampler', '_task_description_str']:
                self._logger_txt.info(f"{attr}: {self._safe_parameter_value(attr, value)}")

        self._logger_txt.info("==================================End of Parameters===============================")
