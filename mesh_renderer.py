import numpy as np
import argparse
import os
from plyfile import PlyData
import mitsuba as mi


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
            <integer name="sampleCount" value="1024"/>
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
    MESH_SEGMENT = """
    <shape type="obj">
        <string name="filename" value="{}"/>
        <transform name="toWorld">
            <scale value="{}"/>
            <rotate x="1" angle="{}"/>
            <rotate y="1" angle="{}"/>
            <rotate z="1" angle="{}"/>
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="twosided">
            <bsdf type="diffuse">
                <rgb name="reflectance" value="0.8,0.8,0.8"/>
            </bsdf>
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


class MeshRenderer:
    XML_HEAD = XMLTemplates.HEAD
    XML_MESH_SEGMENT = XMLTemplates.MESH_SEGMENT
    XML_TAIL = XMLTemplates.TAIL

    def __init__(self, file_path, output_path=None, translation=None, rotation=None):
        self.file_path = file_path
        self.folder, full_filename = os.path.split(file_path)
        self.folder = self.folder or '.'
        self.filename, _ = os.path.splitext(full_filename)
        self.output_path = output_path or self.folder
        self.translation = translation if translation is not None else np.array([0.0, 0.0, 0.0])
        self.rotation = rotation if rotation is not None else np.array([0.0, 0.0, 0.0])


    def load_mesh(self):
        file_extension = os.path.splitext(self.file_path)[1].lower()
        if file_extension == '.obj':
            return self.file_path
        else:
            raise ValueError('Unsupported mesh format. Only .obj files are supported.')
    
    def calculate_mesh_bounds(self, mesh_file_path):
        vertices = []
        with open(mesh_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    coords = line.split()[1:4]
                    vertices.append([float(coord) for coord in coords])
        
        if not vertices:
            raise ValueError('No vertices found in OBJ file')
        
        vertices = np.array(vertices)
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        center = (min_bounds + max_bounds) / 2
        size = max_bounds - min_bounds
        max_dimension = np.max(size)
        
        return center, max_dimension
    
    def normalize_mesh(self, mesh_file_path, target_size=1.0):
        center, max_dimension = self.calculate_mesh_bounds(mesh_file_path)
        
        # Calculate scale to fit mesh in target_size
        scale = target_size / max_dimension if max_dimension > 0 else 1.0
        
        # Calculate translation to center mesh at origin, then apply user translation
        normalization_translation = -center * scale
        final_translation = normalization_translation + self.translation
        
        return scale, final_translation

    def generate_xml_content(self, mesh_file_path, scale=1.0, normalization_translation=None):
        xml_segments = [self.XML_HEAD]
        
        rx_deg = np.degrees(self.rotation[0]) if self.rotation is not None else 0
        ry_deg = np.degrees(self.rotation[1]) if self.rotation is not None else 0
        rz_deg = np.degrees(self.rotation[2]) if self.rotation is not None else 0
        
        # Use normalization translation if provided, otherwise use user translation
        if normalization_translation is not None:
            tx, ty, tz = normalization_translation
        else:
            tx = self.translation[0] if self.translation is not None else 0
            ty = self.translation[1] if self.translation is not None else 0
            tz = self.translation[2] if self.translation is not None else 0
        
        xml_segments.append(self.XML_MESH_SEGMENT.format(
            mesh_file_path, scale, rx_deg, ry_deg, rz_deg, tx, ty, tz
        ))
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

    def process(self):
        from pathlib import Path
        Path(self.output_path).mkdir(exist_ok=True, parents=True)

        mesh_file_path = self.load_mesh()
        
        # Normalize mesh to center and scale appropriately
        scale, normalization_translation = self.normalize_mesh(mesh_file_path, target_size=1.0)
        
        output_filename = self.filename
        output_file_path = f'{self.output_path}/{output_filename}'
        print(f'Processing {output_filename}...')
        
        xml_content = self.generate_xml_content(mesh_file_path, scale, normalization_translation)
        xml_file_path = self.save_xml_content_to_file(output_file_path, xml_content)
        rendered_scene = self.render_scene(xml_file_path)
        self.save_scene(output_file_path, rendered_scene)
        print(f'Finished processing {output_filename}.')


def main():
    parser = argparse.ArgumentParser(description='Render 3D meshes as photorealistic scenes')
    parser.add_argument('filename', help='Path to the mesh file (.obj)')
    parser.add_argument('--output', '-o', help='Output directory path (default: same as input file)')
    parser.add_argument('--translate', nargs=3, type=float, metavar=('x', 'y', 'z'), 
                       help='Translation values for x, y, z axes')
    parser.add_argument('--rotate', nargs=3, type=float, metavar=('rx', 'ry', 'rz'),
                       help='Rotation values for x, y, z axes (in radians)')
    
    args = parser.parse_args()
    
    translation = np.array(args.translate) if args.translate else None
    rotation = np.array(args.rotate) if args.rotate else None
    
    renderer = MeshRenderer(args.filename, args.output, translation, rotation)
    renderer.process()


if __name__ == '__main__':
    main()
