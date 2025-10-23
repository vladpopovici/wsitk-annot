from pathlib import Path
from .annot import Annotation, Dot, Polygon, PointSet, PolyLine
import xmltodict
import xml.etree.ElementTree as ET
from xml.dom import minidom
import openslide as osl

##
## Save annotations to Napari's CSV format.
##
def annotation_to_napari(annot: Annotation, csv_file: str, resolution_mpp: float = -1.0) -> None:
    """Save an <Annotation> in CSV files following Napari's specifications. Note that, since Points
        require a different format from other shapes, a separate file (with suffix '_points') will
        be created for storing all points in the annotation.

        :param annot: (Annotation) an annotation object
        :param csv_file: (str) name of the file for saving the annotation
        :param resolution_mpp: (float) if positive then the desired resolution (microns-per-pixel)
            for the saved annotation (all is scaled by accordingly).
        :param layer: (str) name of the layer from which to save the annotation

        :return: True if everything is OK
        """

    # scale the annotation for the target:
    if resolution_mpp > 0.0:
        annot.set_resolution(resolution_mpp)

    points_file = Path(csv_file).with_name(Path(csv_file).stem + '_points.csv')
    shapes_file = Path(csv_file)

    points_lines = []
    points_idx = 0
    shapes_lines = []
    shapes_idx = 0

    for a in annot._annots:
        if a._annotation_type == "DOT":
            points_lines.append([points_idx, a.y, a.x])
            points_idx += 1
        elif a._annotation_type in ["POLYGON", "CIRCLE"]:
            xy = a.xy()
            for k in np.arange(xy.shape[0]):
                shapes_lines.append([shapes_idx, 'polygon', k, xy[k, 1], xy[k, 0]])
            shapes_idx += 1
        elif a._annotation_type == "POINTSET":
            xy = a.xy()
            for k in np.arange(xy.shape[0]):
                shapes_lines.append([shapes_idx, 'path', k, xy[k, 1], xy[k, 0]])
            shapes_idx += 1
        else:
            raise RuntimeWarning("not implemented annotation type " + a._annotation_type)

    if points_idx > 0:
        # more than the header
        with open(points_file, 'w') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONE)
            writer.writerow(['index', 'axis-0', 'axis-1'])
            writer.writerows(points_lines)

    if shapes_idx > 0:
        # more than the header
        with open(shapes_file, 'w') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONE)
            writer.writerow(['index', 'shape-type', 'vertex-index', 'axis-0', 'axis-1'])
            writer.writerows(shapes_lines)

    return

##
## Import/export functions for interoperability with other formats.
##
def annotation_from_ASAP(
        infile: Union[str|Path],
        wsi_extent: dict[int, int],
        mpp: float
) -> Annotation:
    """
    Import annotations from ASAP (XML) files. See also
    https://github.com/computationalpathologygroup/ASAP
    """
    infile = Path(infile)
    with open(infile, 'r') as inp:
        annot_dict = xmltodict.parse(inp.read(), xml_attribs=True)

    if 'ASAP_Annotations' not in annot_dict:
        raise RuntimeError('Syntax error in ASAP XML file')

    annot_dict = annot_dict['ASAP_Annotations']
    if 'Annotations' not in annot_dict:
        raise RuntimeError('Syntax error in ASAP XML file')

    annot_list = annot_dict['Annotations']['Annotation']  # many nested levels...

    wsi_annotation = Annotation(infile.name, wsi_extent, mpp)
    for annot in annot_list:
        if annot['@Type'].lower() == 'dot':
            coords = [float(annot["Coordinates"]["Coordinate"]["@X"]), float(annot["Coordinates"]["Coordinate"]["@Y"])]
            obj = Dot(coords, annot['@Name'])
        elif annot['@Type'].lower() == 'pointset':
            coords = [(float(o["@X"]), float(o["@Y"])) for o in annot["Coordinates"]["Coordinate"]]
            obj = PointSet(coords, annot['@Name'])
        elif annot['@Type'].lower() == 'polygon':
            coords = [(float(o["@X"]), float(o["@Y"])) for o in annot["Coordinates"]["Coordinate"]]
            obj = Polygon(coords, annot['@Name'])
        else:
            raise RuntimeError(f"Unknown annotation type {annot['@Type']}")
        obj.set_property('group', [annot["@PartOfGroup"]])
        wsi_annotation.add_annotation_object(obj)

    return wsi_annotation


# def ndpa_read(ndpa_file, force_closed=False):
#     """Read all annotations.

#     Args:
#         ndpa_file (str): annotation file name
#         force_closed (bool): should polygonal lines be forced to form a closed
#             contour (i.e. first and last points are identical)

#     Returns:
#         -a dictionary with keys corresponding to annotation object
#         names and with values the corresponding lists of points

#     See also:
#         ndpa_read_single
#     """


#     xml_file = ET.parse(ndpa_file)
#     xml_root = xml_file.getroot()

#     annot = dict()

#     for ann in list(xml_root):
#         name = ann.find('title').text
#         annot[name] = []

#         p = ann.find('annotation')
#         if p is None:
#             continue

#         p = p.find('pointlist')
#         if p is None:
#             continue

#         coords = np.array([(np.long(pts.find('x').text), np.long(pts.find('y').text))
#                            for pts in list(p)])
#         if np.all(coords[0,] == coords[-1,]):
#             # actually, it's a polygon
#             annot[name].append(Polygon(coords, name=name))
#         else:
#             # it's a set of points, but may be forced to Polygon
#             if force_closed:
#                 coords = np.vstack((coords, coords[0,]))
#                 annot[name].append(Polygon(coords, name=name))
#             else:
#                 annot[name].append(PointSet(coords, name=name))

#     return annot
# ##-

# def annotation_from_NDPA(
#         infile: Union[str|Path],
#         wsi_extent: dict[int, int],
#         mpp: float
# ) -> Annotation:
#     """
#     Import annotations from NDPA (XML) files produced by Hammamatsu's NDP.view software.
#     """
#     infile = Path(infile)
#     xml_file = ET.parse(infile)
#     xml_root = xml_file.getroot()
    
#     ndpi = osl.OpenSlide(args.ndpi)
#     x_off = int(ndpi.properties['hamamatsu.XOffsetFromSlideCentre'])
#     y_off = int(ndpi.properties['hamamatsu.YOffsetFromSlideCentre'])
#     x_mpp = float(ndpi.properties['openslide.mpp-x'])
#     y_mpp = float(ndpi.properties['openslide.mpp-y'])
#     dimX0, dimY0 = ndpi.level_dimensions[0]
    
    
#     for ann in list(xml_root):
#         name = ann.find('title').text
#         p = ann.find('annotation')
#         if p is None:
#             continue
#         p = p.find('pointlist')
#         if p is None:
#             continue
        
#         for pts in list(p):
#             # coords in NDPI system, relative to the center of the slide
#             x = int(pts.find('x').text)
#             y = int(pts.find('y').text)
            
#             # convert the coordinates:
#             x -= x_off                 # relative to the center of the image
#             y -= y_off
            
#             x /= (1000*x_mpp)          # in pixels, relative to the center
#             y /= (1000*y_mpp)
            
#             x = int(x + dimX0 / 2)    # in pixels, relative to UL corner
#             y = int(y + dimY0 / 2)
            
#             # save the coordinates
#             fout.write('\t'.join([str(int(x/d)), str(int(y/d)), '\n']))
#             #print(int(x/d), int(y/d))
        
#         fout.close()
