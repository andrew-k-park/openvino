# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import gc

import numpy as np
from models_hub_common.multiprocessing_utils import multiprocessing_run
from openvino.runtime import Core
from openvino.tools.mo import convert_model


class TestConvertModel:
    infer_timeout = 600

    def load_model(self, model_name, model_link):
        raise "load_model is not implemented"

    def get_inputs_info(self, model_obj):
        raise "get_inputs_info is not implemented"

    def prepare_input(self, input_shape, input_type):
        if input_type in [np.float32, np.float64]:
            return np.random.randint(-2, 2, size=input_shape).astype(input_type)
        elif input_type in [np.int8, np.int16, np.int32, np.int64]:
            return np.random.randint(-5, 5, size=input_shape).astype(input_type)
        elif input_type in [np.uint8, np.uint16]:
            return np.random.randint(0, 5, size=input_shape).astype(input_type)
        elif input_type in [str]:
            return np.broadcast_to("Some string", input_shape)
        elif input_type in [bool]:
            return np.random.randint(0, 2, size=input_shape).astype(input_type)
        else:
            assert False, "Unsupported type {}".format(input_type)

    def prepare_inputs(self, inputs_info):
        inputs = []
        for input_shape, input_type in inputs_info:
            inputs.append(self.prepare_input(input_shape, input_type))
        return inputs

    def convert_model(self, model_obj):
        ov_model = convert_model(model_obj)
        return ov_model

    def infer_fw_model(self, model_obj, inputs):
        raise "infer_fw_model is not implemented"

    def infer_ov_model(self, ov_model, inputs, ie_device):
        core = Core()
        compiled = core.compile_model(ov_model, ie_device)
        ov_outputs = compiled(inputs)
        return ov_outputs

    def compare_results(self, fw_outputs, ov_outputs):
        assert len(fw_outputs) == len(ov_outputs), \
            "Different number of outputs between TensorFlow and OpenVINO:" \
            " {} vs. {}".format(len(fw_outputs), len(ov_outputs))

        fw_eps = 5e-2
        is_ok = True
        for out_name in fw_outputs.keys():
            cur_fw_res = fw_outputs[out_name]
            assert out_name in ov_outputs, \
                "OpenVINO outputs does not contain tensor with name {}".format(out_name)
            cur_ov_res = ov_outputs[out_name]
            print(f"fw_re: {cur_fw_res};\n ov_res: {cur_ov_res}")
            if not np.allclose(cur_ov_res, cur_fw_res,
                               atol=fw_eps,
                               rtol=fw_eps, equal_nan=True):
                is_ok = False
                print("Max diff is {}".format(np.array(abs(cur_ov_res - cur_fw_res)).max()))
            else:
                print("Accuracy validation successful!\n")
                print("absolute eps: {}, relative eps: {}".format(fw_eps, fw_eps))
        assert is_ok, "Accuracy validation failed"

    def teardown_method(self):
        # deallocate memory after each test case
        gc.collect()

    def _run(self, model_name, model_link, ie_device):
        print("Load the model {} (url: {})".format(model_name, model_link))
        fw_model = self.load_model(model_name, model_link)
        print("Retrieve inputs info")
        inputs_info = self.get_inputs_info(fw_model)
        print("Prepare input data")
        inputs = self.prepare_inputs(inputs_info)
        print("Convert the model into ov::Model")
        ov_model = self.convert_model(fw_model)
        print("Infer the original model")
        fw_outputs = self.infer_fw_model(fw_model, inputs)
        print("Infer ov::Model")
        ov_outputs = self.infer_ov_model(ov_model, inputs, ie_device)
        print("Compare TensorFlow and OpenVINO results")
        self.compare_results(fw_outputs, ov_outputs)

    def run(self, model_name, model_link, ie_device):
        multiprocessing_run(self._run, [model_name, model_link, ie_device], model_name, self.infer_timeout)
