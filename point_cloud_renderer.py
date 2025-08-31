import argparse
import os
from pathlib import Path

import mitsuba as mi
import numpy as np
from plyfile import PlyData


class XMLTemplates:
    HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1920"/>
            <integer name="height" value="1080"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
"""
    BALL_SEGMENT = """
    <shape type="sphere">
        <float name="radius" value="0.015"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""
    TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""


class PointCloudRenderer:
    POINTS_PER_OBJECT = 2048
    XML_HEAD = XMLTemplates.HEAD
    XML_BALL_SEGMENT = XMLTemplates.BALL_SEGMENT
    XML_TAIL = XMLTemplates.TAIL

    def __init__(self, file_path, output_path=None, translation=None, rotation=None):
        self.file_path = file_path
        self.folder, full_filename = os.path.split(file_path)
        self.folder = self.folder or '.'
        self.filename, _ = os.path.splitext(full_filename)
        self.output_path = output_path or self.folder
        self.translation = translation if translation is not None else np.array([0.0, 0.0, 0.0])
        self.rotation = rotation if rotation is not None else np.array([0.0, 0.0, 0.0])

    @staticmethod
    def compute_color(x, y, z):
        vec = np.clip(np.array([x, y, z]), 0.001, 1.0)
        vec /= np.linalg.norm(vec)
        return vec

    @staticmethod
    def standardize_point_cloud(pcl, points_per_object=POINTS_PER_OBJECT):
        pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
        pcl = pcl[pt_indices]
        center = np.mean(pcl, axis=0)
        scale = np.amax(pcl - np.amin(pcl, axis=0))
        return ((pcl - center) / scale).astype(np.float32) * 0.5

    @staticmethod
    def apply_rotation(pcl, rotation):
        rx, ry, rz = rotation
        
        # Rotation matrix around X-axis
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        # Rotation matrix around Y-axis  
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        # Rotation matrix around Z-axis
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        # Combined rotation matrix (order: Z * Y * X)
        R = Rz @ Ry @ Rx
        
        return (R @ pcl.T).T

    @staticmethod
    def apply_translation(pcl, translation):
        return pcl + translation

    def load_point_cloud(self):
        file_extension = os.path.splitext(self.file_path)[1]
        if file_extension == '.npy':
            return np.load(self.file_path, allow_pickle=True)
        elif file_extension == '.npz':
            return np.load(self.file_path)['pred']
        elif file_extension == '.ply':
            ply_data = PlyData.read(self.file_path)
            return np.column_stack(ply_data['vertex'][t] for t in ('x', 'y', 'z'))
        else:
            raise ValueError('Unsupported file format.')

    def generate_xml_content(self, pcl, pcl_colors = None):
        xml_segments = [self.XML_HEAD]
        for i, point in enumerate(pcl):
            if pcl_colors is not None:
                color = pcl_colors[i]
            else:
                color = self.compute_color(
                    point[0] + 0.5, point[1] + 0.5, point[2] + 0.5 - 0.0125)
            xml_segments.append(self.XML_BALL_SEGMENT.format(
                point[0], point[1], point[2], *color))
        xml_segments.append(self.XML_TAIL)
        return ''.join(xml_segments)

    @staticmethod
    def save_xml_content_to_file(output_file_path, xml_content):
        xml_file_path = f'{output_file_path}.xml'
        with open(xml_file_path, 'w') as f:
            f.write(xml_content)
        return xml_file_path

    @staticmethod
    def render_scene(xml_file_path):
        mi.set_variant('scalar_rgb')
        scene = mi.load_file(xml_file_path)
        img = mi.render(scene)
        return img

    @staticmethod
    def save_scene(output_file_path, rendered_scene):
        mi.util.write_bitmap(f'{output_file_path}.png', rendered_scene)

    def render_pcl_and_color(self, pcl, color, output_file_path):
        """Renders a point cloud with optional color info.
        
        Args:
            pcl: Nx3 point cloud.
            color: Nx3 RGB color array, or None in which case x,y,z colormap is generated.
            output_file_path: (Temp) file path to save XML config to.
        
        Returns:
            Image as a HxWx3 numpy array.
        """
        pcl = self.standardize_point_cloud(pcl)
        pcl = self.apply_rotation(pcl, self.rotation)
        pcl = self.apply_translation(pcl, self.translation)
        pcl = pcl[:, [2, 0, 1]]  # TODO: not sure why we do this.

        xml_content = self.generate_xml_content(pcl, color)
        xml_file_path = self.save_xml_content_to_file(output_file_path, xml_content)
        return self.render_scene(xml_file_path)

    def process(self):
        if self.output_path is not None and len(self.output_path) > 0:
            save_dir = Path(self.output_path).expanduser().resolve()
            save_dir.mkdir(exist_ok=True, parents=True)
        else:
            save_dir = Path(".").resolve()

        pcl_data = self.load_point_cloud()
        if len(pcl_data.shape) < 3:
            pcl_data = pcl_data[np.newaxis, :, :]

        for index, pcl in enumerate(pcl_data):
            pcl = self.standardize_point_cloud(pcl)
            pcl = self.apply_rotation(pcl, self.rotation)
            pcl = self.apply_translation(pcl, self.translation)
            pcl = pcl[:, [2, 0, 1]]

            output_filename = f'{self.filename}_{index:02d}'
            output_file_path = save_dir / output_filename
            print(f'Processing {output_filename}...')
            rendered_scene = self.render_pcl_and_color(pcl, None, output_file_path)
            self.save_scene(output_file_path, rendered_scene)
            print(f'Finished processing {output_filename}.')


def main():
    parser = argparse.ArgumentParser(description='Render point clouds as 3D scenes')
    parser.add_argument('filename', help='Path to the point cloud file (.npy, .npz, or .ply)')
    parser.add_argument('--output', '-o', help='Output directory path (default: same as input file)')
    parser.add_argument('--translate', nargs=3, type=float, metavar=('x', 'y', 'z'), 
                       help='Translation values for x, y, z axes')
    parser.add_argument('--rotate', nargs=3, type=float, metavar=('rx', 'ry', 'rz'),
                       help='Rotation values for x, y, z axes (in radians)')
    
    args = parser.parse_args()
    
    translation = np.array(args.translate) if args.translate else None
    rotation = np.array(args.rotate) if args.rotate else None
    
    renderer = PointCloudRenderer(args.filename, args.output, translation, rotation)
    renderer.process()


if __name__ == '__main__':
    main()
