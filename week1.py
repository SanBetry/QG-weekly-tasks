import json
import numpy as np
import math

#初始化方法
class coordinate_system:
    def __init__(self, ori_axis, vectors): #self表示当前坐标系  
        self.original_base = np.array(ori_axis)
        self.original_vectors = np.array(vectors)
        self.base = self.original_base
        self.vectors = self.original_vectors

    #重置坐标系
    def reset(self):
        self.base = self.original_base
        self.vectors = self.original_vectors
    #投影计算

    def projection(self):
        proj = []
        for vec in self.vectors: #遍历每个要运算的向量
            single_vec_proj = []
            for axis in self.base:
                axis_norm = np.linalg.norm(axis).item()
                if axis_norm == 0:
                    single_vec_proj.append(0.0)
                    continue
                dot_prod = np.dot(vec, axis).item()
                single_vec_proj.append(dot_prod / axis_norm)
            proj.append(single_vec_proj)
        return proj

    def angle(self):
        angles = []
        for vec in self.vectors:
            vec_norm = np.linalg.norm(vec)
            vec_angle = []
            for axis in self.base:
                axis_norm = np.linalg.norm(axis)
                dot_product = np.dot(vec, axis).item() #为什么要有个点？
                norm_product = (axis_norm * vec_norm).item()
                if norm_product == 0:
                    vec_angle.append(0.0)
                    continue
                cos_theta = dot_product / norm_product
                theta = math.acos(np.clip(cos_theta, -1.0, 1.0))
                vec_angle.append(theta)
            angles.append(vec_angle)
        return angles

    def area (self):
        return abs(np.linalg.det(self.base.T))

    def scale (self):
        try:
            det = np.linalg.det(self.base.T)
            return abs(det)
        except:
            return "维度不匹配，无法计算行列式"

    def transform(self, obj_axis):
        obj_base = np.array(obj_axis) #目标坐标系基向量
        try:
            inv_base = np.linalg.inv(obj_base.T) #目标基矩阵转逆
            trans_matrix =inv_base @ self.base.T
            new_vec = trans_matrix @ self.vectors.T
            self.base = obj_base #更新坐标系为目标坐标系
            self.vectors = new_vec.T
            return self.vectors.tolist() #NumPy数组->Python列表
        except:
            return "坐标系不合法"

def process_tasks(json_path):
    # python打开文件的写法
    with open(json_path, "r", encoding="utf-8") as f:
        task_groups = json.load(f)

    for idx, group in enumerate(task_groups):
        group_name = group["group_name"]
        print(f"\n 处理第{idx + 1}组数据：{group_name}")

        ori_axis = group["ori_axis"]
        vectors = group["vectors"]
        tasks = group["tasks"]

        cs = coordinate_system(ori_axis, vectors)

        for task in tasks:
            task_type = task["type"]
            if task_type == "axis_projection":
                print(f"--投影结果为：{cs.projection()}")
            elif task_type == "axis_angle":
                print(f"--夹角为：{cs.angle()}")
            elif task_type == "area":
                print(f"--缩放为：{cs.scale()}")
            elif task_type == "change_axis":
                obj_axis = task["obj_axis"]
                print(f"--坐标转移结果为：{cs.transform(obj_axis)}")


if __name__ == "__main__":
    json_path = "data(1).json"
    process_tasks(json_path)
