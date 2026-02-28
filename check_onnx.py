from onnx.checker import check_graph, check_attribute, check_model, check_node, check_value_info

import onnx
import os
import sys
import argparse

import logging

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("./logs/check_onnx.log", mode="w", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

default_model_path = "./models/voxcpm_1.5_onnx/"

def set_graph_names(graph, base_name="graph"):
    if not graph.name:
        graph.name = base_name
    # 遍历所有节点，检查是否有子图
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                # 递归处理子图
                set_graph_names(attr.g, base_name + "_sub")
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    set_graph_names(g, base_name + "_sub")

def get_graph_names(graph, names=None):
    if names is None:
        names = set()
    if graph.name:
        names.add(graph.name)
    # 遍历所有节点，检查是否有子图
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                # 递归处理子图
                get_graph_names(attr.g, names)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    get_graph_names(g, names)
                    
    logger.info(f"Graph '{graph.name}' has subgraphs: {names}")
    return names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check ONNX model.")
    parser.add_argument("-m", "--model_path", type=str, default=default_model_path, help="Path to the ONNX model.")
    parser.add_argument("-c", "--simple_check", action="store_true", help="Only check onnx model with simple mode.")
    args = parser.parse_args()
    
    model_file_path = [os.path.join(args.model_path, f) for f in os.listdir(args.model_path) if f.endswith(".onnx")]
    
    # check onnx model
    try:
        logger.info(f"Checking ONNX model in {args.model_path}...")
        for onx_file in model_file_path:
            onnx_proto = onnx.load(onx_file, load_external_data=True)
            
            # calculate size of onnx model file and its external data
            onnx_size = os.path.getsize(onx_file) / (1024 * 1024)
            
            external_files = set()
            for init in onnx_proto.graph.initializer:
                if len(init.external_data) > 0:
                    logger.info(f"Tensor '{init.name}' has external data: {init.external_data}")
                    # external_data 是一个 StringStringEntryProto 列表
                    for entry in init.external_data:
                        if entry.key == "location":
                            file_name = entry.value
                            # 获取绝对路径（假设外部文件与主模型在同一目录）
                            full_path = os.path.join(os.path.dirname(onx_file), file_name)
                            external_files.add(full_path)
                            logger.info(f"Tensor '{init.name}' uses external file: {full_path}")
            
            # calculate size of external data
            external_data_size = sum(os.path.getsize(f) for f in external_files) / (1024 * 1024)
            
            total_size = onnx_size + external_data_size
            
            logger.info(f"{onx_file} model size: {onnx_size:.2f} MB ---- {onnx_size / 1024:.2f} GB")
            logger.info(f"{onx_file} external data size: {external_data_size:.2f} MB ---- {external_data_size / 1024:.2f} GB")
            logger.info(f"{onx_file} total size: {total_size:.2f} MB ---- {total_size / 1024:.2f} GB")
            
            get_graph_names(onnx_proto.graph)
            external_count = sum(1 for init in onnx_proto.graph.initializer if init.external_data)
            logger.info(f"Checking ONNX model {onx_file}...")
            logger.info(f"External initializer num: {external_count}")
            logger.info(f"{onx_file} model info:\n model version: {onnx_proto.model_version}\n ir version: {onnx_proto.ir_version}\n op version: {onnx_proto.opset_import[0].version}\n producer name: {onnx_proto.producer_name}\n producer version: {onnx_proto.producer_version}\n domain: {onnx_proto.domain}\n doc string: {onnx_proto.doc_string}\n graph name: {onnx_proto.graph.name}\n graph doc string: {onnx_proto.graph.doc_string}\n graph input num: {len(onnx_proto.graph.input)}\n graph output num: {len(onnx_proto.graph.output)}\n graph node num: {len(onnx_proto.graph.node)}\n graph initializer num: {len(onnx_proto.graph.initializer)}")
            
            if total_size / 1024 < 2 and not args.simple_check:
                # check onnx graph
                check_graph(onnx_proto.graph)
                
                # check onnx model
                check_model(onnx_proto, full_check=True)
                
                # check onnx attr
                for node in onnx_proto.graph.node:
                    for attr in node.attribute:
                        check_attribute(attr)
                
                # check onnx node
                for node in onnx_proto.graph.node:
                    check_node(node)
                    
                # check onnx value info
                for value_info in onnx_proto.graph.value_info:
                    check_value_info(value_info)
            else:
                logger.warning(f"Model {onx_file} is too large to check (total size: {total_size:.2f} MB). Skipping detailed checks to avoid potential memory issues.")
                check_model(onx_file, full_check=True)
                
    except Exception as e:
        logger.error(f"ONNX model check failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
        
    logger.info("ONNX model check passed successfully.")