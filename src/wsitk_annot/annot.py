# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.2

## This module handles whole slide image annotations for own algorithms as well
## as several import/export formats (HistomicsTK, Hamamatsu, ASAP, etc).
##
## The annotation objects belong to one group, at least, named "no_group".
## Other groups may be added, and objects may belong to several groups.
##
## All annotations share the same underlying mesh (= a raster of pixels with
## predefined extent and fixed resolution (microns-per-pixels)).

__all__ = ['AnnotationObject', 'Dot', 'Polygon', 'PointSet', 'Annotation', 'Circle']

import csv
from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path
import shapely
import shapely.geometry as shg
import shapely.affinity as sha
import geojson as gj
import xmltodict
import numpy as np
import collections
from ._serialize import pack as j_pack, unpack as j_unpack

##-
class AnnotationObject(ABC):
    """Define the AnnotationObject minimal interface. This class is made
    abstract to force more meaningful names (e.g. Dot, Polygon, etc.) in
    subclasses."""

    def __init__(self):
        # main geometrical object describing the annotation:

        self._geom = shg.base.BaseGeometry()
        self._name = None
        self._annotation_type = None
        self._metadata = {'group': ['no_group']}

    @abstractmethod
    def duplicate(self):
        pass

    def __str__(self):
        """Return a string representation of the object."""
        return str(self.type) + " <" + str(self.name) + ">: " + str(self.geom)

    def bounding_box(self):
        """Compute the bounding box of the object."""
        return self.geom.bounds

    def translate(self, x_off, y_off=None):
        """Translate the object by a vector [x_off, y_off], i.e.
        the new coordinates will be x' = x + x_off, y' = y + y_off.
        If y_off is None, then the same value as in x_off will be
        used.

        :param x_off: (double) shift in thr X-direction
        :param y_off: (double) shift in the Y-direction; if None,
            y_off == x_off
        """
        if y_off is None:
            y_off = x_off
        self._geom = sha.translate(self.geom, x_off, y_off, zoff=0.0)

        return

    def scale(self, x_scale, y_scale=None, origin='center'):
        """Scale the object by a specified factor with respect to a specified
        origin of the transformation. See shapely.geometry.scale() for details.

        :param x_scale: (double) X-scale factor
        :param y_scale: (double) Y-scale factor; if None, y_scale == x_scale
        :param origin: reference point for scaling. Default: "center" (of the
            object). Alternatives: "centroid" or a shapely.geometry.Point object
            for arbitrary origin.
        """
        if y_scale is None:
            y_scale = x_scale
        self._geom = sha.scale(self.geom, xfact=x_scale, yfact=y_scale, zfact=1, origin=origin)

        return

    def resize(self, factor: float) -> None:
        """Resize an object with the specified factor. This is equivalent to
        scaling with the origin set to (0,0) and same factor for both x and y
        coordinates.

        :param factor: (float) resizing factor.
        """
        self.scale(factor, origin=shg.Point((0.0, 0.0)))

        return

    def affine(self, M):
        """Apply an affine transformation to all points of the annotation.

        If M is the affine transformation matrix, the new coordinates
        (x', y') of a point (x, y) will be computed as

        x' = M[1,1] x + M[1,2] y + M[1,3]
        y' = M[2,1] x + M[2,2] y + M[2,3]

        In other words, if P is the 3 x n matrix of n points,
        P = [x; y; 1]
        then the new matrix Q is given by
        Q = M * P

        :param M: numpy array [2 x 3]

        :return:
            nothing
        """

        self._geom = sha.affine_transform(self.geom, [M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 2], M[1, 2]])

        return

    @property
    def geom(self):
        """The geometry of the object."""
        return self._geom

    @property
    def group(self) -> list:
        """Return the name of the annotation group."""
        return self._metadata['group']

    @property
    def name(self) -> str:
        """Return the name of the annotation."""
        return self._name

    @property
    def type(self):
        """Return the annotation type as a string."""
        return self._annotation_type

    def get_property(self, property_name: str):
        return self._metadata[property_name]

    def set_property(self, property_name: str, value):
        self._metadata[property_name] = value

    def x(self):
        """Return the x coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return shapely.get_coordinates(self.geom)[:,0]

    def y(self):
        """Return the y coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return shapely.get_coordinates(self.geom)[:,1]

    def xy(self) -> np.array:
        return shapely.get_coordinates(self.geom)

    def size(self) -> int:
        """Return the number of points defining the object."""
        raise NotImplementedError

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        raise NotImplementedError

    def fromdict(self, d: dict) -> None:
        """Intialize the objct from a dictionary."""
        raise NotImplementedError

    def asGeoJSON(self) -> dict:
        """Return a dictionary compatible with GeoJSON specifications."""
        return gj.Feature(geometry=shg.mapping(self.geom),
                          properties=dict(object_type="annotation",
                                          annotation_type=self._annotation_type,
                                          name=self._name,
                                          metadata=self._metadata)
                          )

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        """This is a basic function - further tests should be implemented for particular object
        types."""

        self._geom = shg.shape(d["geometry"])
        try:
            self._name = d["properties"]["name"]
            self._metadata = d["properties"]["metadata"]
        except KeyError:
            pass
##-


##-
class Dot(AnnotationObject):
    """Dot: a single position in the image."""

    def __init__(self, coords=(0.0, 0.0), name=None):
        """Initialize a DOT annotation, i.e. a single point in plane.

        Args:
            coords (list or vector or tuple): the (x,y) coordinates of the point
            name (str): the name of the annotation
        """
        super().__init__()

        self._annotation_type = "DOT"
        self._name = "DOT" if name is None else name

        if not isinstance(coords, collections.Iterable):
            raise RuntimeError('coords parameter cannot be interpreted as a 2D vector')

        self._geom = shg.Point(coords)

        return

    def duplicate(self):
        return Dot(shapely.get_coordinates(self.geom), name=self.name)

    def size(self) -> int:
        """Return the number of points defining the object."""
        return 1

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        d = {
            "annotation_type": self._annotation_type,
            "name": self._name,
            "x": self.x(),
            "y": self.y(),
            "metadata": self._metadata,
        }

        return d

    def fromdict(self, d: dict) -> None:
        """Initialize the object from a dictionary."""

        self._annotation_type = d["annotation_type"]
        self._name = d["name"]
        self._geom = shg.Point((d["x"], d["y"]))
        self._metadata = d["metadata"]

        return

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        if d["geometry"]["type"].lower() != "point":
            raise RuntimeError("Need a Point feature! Got: " + str(d))

        super().fromGeoJSON(d)
        self._annotation_type = "DOT"

        return
##-


##-
class PointSet(AnnotationObject):
    """PointSet: an ordered collection of points."""

    def __init__(self, coords: Union[list|tuple], name=None):
        """Initialize a POINTSET annotation, i.e. a collection
         of points in plane.

        Args:
            coords (list or tuple): coordinates of the points as in [(x0,y0), (x1,y1), ...]
            name (str): the name of the annotation
        """

        super().__init__()
        self._annotation_type = "POINTSET"
        self._name = "POINTS" if name is None else name

        # check whether coords is iterable and build the coords from it:
        if not isinstance(coords, collections.Iterable):
            raise RuntimeError('coords parameter cannot be interpreted as a 2D array')

        self._geom = shg.MultiPoint(coords)

        return

    def duplicate(self):
        return PointSet(shapely.get_coordinates(self.geom), name=self.name)

    def size(self) -> int:
        """Return the number of points defining the object."""
        return self.xy().shape[0]

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        d = {
            "annotation_type": self._annotation_type,
            "name": self._name,
            "x": self.x(),
            "y": self.y(),
            "metadata": self._metadata,
        }

        return d

    def fromdict(self, d: dict) -> None:
        """Initialize the object from a dictionary."""

        self._annotation_type = d["annotation_type"]
        self._name = d["name"]
        self._geom = shg.MultiPoint(zip(d["x"], d["y"]))
        self._metadata = d["metadata"]

        return

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        if d["geometry"]["type"].lower() != "multipoint":
            raise RuntimeError("Need a MultiPoint feature! Got: " + str(d))

        super().fromGeoJSON(d)
        self._annotation_type = "POINTSET"

        return
##-


class PolyLine(AnnotationObject):
    """PolyLine: polygonal line (a sequence of segments)"""

    def __init__(self, coords, name=None):
        """Initialize a POLYLINE object.

        Args:
            coords (list or tuple): coordinates of the points [(x0,y0), (x1,y1), ...]
                defining the segments (x0,y0)->(x1,y1); (x1,y1)->(x2,y2),...
            name (str): the name of the annotation
        """
        super().__init__()
        self._annotation_type = "POLYLINE"
        self._name = name if not None else "POLYLINE"

        # check whether x is iterable and build the coords from it:
        if not isinstance(coords, collections.Iterable):
            raise RuntimeError('coords parameter cannot be interpreted as a 2D array')

        self._geom = shg.LineString(coords)

        return

    def duplicate(self):
        return PolyLine(shapely.get_coordinates(self.geom), name=self.name)

    def size(self) -> int:
        """Return the number of points defining the object."""
        return self.xy().shape[0]

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        d = {
            "annotation_type": self._annotation_type,
            "name": self._name,
            "x": self.x(),
            "y": self.y(),
            "metadata": self._metadata,
        }

        return d

    def fromdict(self, d: dict) -> None:
        """Initialize the object from a dictionary."""

        self._annotation_type = "POLYLINE"
        self._name = d["name"]
        self._geom = shg.LineString(zip(d["x"], d["y"]))
        self._metadata = d["metadata"]

        return

    def asGeoJSON(self) -> dict:
        """Return a dictionary compatible with GeoJSON specifications."""
        return gj.Feature(geometry=gj.LineString(zip(self.x(), self.y())),
                          properties=dict(object_type="annotation",
                                          annotation_type=self._annotation_type,
                                          name=self._name,
                                          metadata=self._metadata)
                          )

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        if d["geometry"]["type"].lower() != "linestring":
            raise RuntimeError("Need a LineString feature! Got: " + str(d))

        super().fromGeoJSON(d)
        self._annotation_type = "POLYLINE"

        return
##-

##-
class Polygon(AnnotationObject):
    """Polygon: an ordered collection of points where the first and
    last points coincide."""

    def __init__(self, coords, name=None):
        """Initialize a POINTSET annotation, i.e. a collection
         of points in plane.

        Args:
            coords (list or tuple): coordinates of the points as in [(x0,y0), (x1,y1), ...]
            name (str): the name of the annotation
        """

        super().__init__()
        self._annotation_type = "POLYGON"
        self._name = name if not None else "POLYGON"

        # check whether x is iterable and build the coords from it:
        if not isinstance(coords, collections.Iterable):
            raise RuntimeError('coords parameter cannot be interpreted as a 2D array')

        self._geom = shg.Polygon(coords)

        return

    def duplicate(self):
        return Polygon(shapely.get_coordinates(self.geom), name=self.name)

    def size(self) -> int:
        """Return the number of points defining the object."""
        return self.xy().shape[0]

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        d = {
            "annotation_type": self._annotation_type,
            "name": self._name,
            "x": self.x(),
            "y": self.y(),
            "metadata": self._metadata,
        }

        return d

    def fromdict(self, d: dict) -> None:
        """Initialize the object from a dictionary."""

        self._annotation_type = d["annotation_type"]
        self._name = d["name"]
        self._geom = shg.Polygon(zip(d["x"], d["y"]))
        self._metadata = d["metadata"]

        return

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        if d["geometry"]["type"].lower() != "polygon":
            raise RuntimeError("Need a Polygon feature! Got: " + str(d))

        super().fromGeoJSON(d)
        self._annotation_type = "POLYGON"

        return
##-

##
class Circle(Polygon):
    """Circle annotation is implemented as a polygon (octogon) to be compatible with GeoJSON specifications."""
    def __init__(self, center, radius, name=None):
        alpha = np.array([k*np.pi/4 for k in range(8)])
        coords = np.vstack((
            radius * np.sin(alpha) + center[0], radius * np.cos(alpha) + center[1]
        )).transpose()

        super().__init__(coords, name)
        self._annotation_type = "CIRCLE"

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        if d["geometry"]["type"].lower() != "circle":
            raise RuntimeError("Need a CIRCLE feature! Got: " + str(d))

        super().fromGeoJSON(d)
        self._annotation_type = "CIRCLE"

        return
##-


def createEmptyAnnotationObject(annot_type: str) -> AnnotationObject:
    """Function to create an empty annotation object of a desired type.

    Args:
        annot_type (str):
            type of the annotation object:
            DOT/POINT
            POINTSET
            POLYLINE/LINESTRING
            POLYGON
            CIRCLE

    """

    obj = None
    if annot_type.upper() == 'DOT' or annot_type.upper() == 'POINT':
        obj = Dot(coords=[0, 0])
    elif annot_type.upper() == 'POINTSET':
        obj = PointSet([[0, 0]])
    elif annot_type.upper() == 'LINESTRING' or annot_type.upper() == 'POLYLINE':
        obj = PolyLine([[0, 0], [1, 1], [2, 2]])
    elif annot_type.upper() == 'POLYGON':
        obj = Polygon([[0, 0], [1, 1], [2, 2]])
    elif annot_type.upper() == 'CIRCLE':
        obj = Circle([0.0, 0.0], 1.0)
    else:
        raise RuntimeError("unknown annotation type: " + annot_type)
    return obj


##-
class Annotation(object):
    """
    An annotation is a collection of AnnotationObjects represented on the same
    coordinate system (mesh).
    """

    def __init__(self, name: str, image_shape: dict, mpp: float) -> None:
        """Initialize an Annotation for a slide.

        :param name: (str) name of the annotation
        :param image_shape: (dict) shape of the image corresponding to the annotation
            {'width':..., 'height':...}
        :param mpp: (float) slide resolution (microns-per-pixel) for the image
        """
        self._name = name
        self._image_shape = dict(width=0, height=0)
        self._annots = []
        self._mpp = mpp

        if 'width' not in image_shape or 'height' not in image_shape:
            raise RuntimeError('Invalid shape specification (<width> or <height> key missing)')

        self._image_shape = image_shape

        return

    def add_annotation_object(self, a: AnnotationObject) -> None:
        self._annots.append(a)

    def add_annotations(self, a: list) -> None:
        self._annots.extend(a)

    def get_base_image_shape(self) -> dict:
        return self._image_shape

    @property
    def name(self):
        """Return the name of the annotation object."""
        return self._name

    @property
    def type(self):
        """Return the annotation type as a string."""
        return 'Annotation'

    def get_resolution(self) -> float:
        return self._mpp

    def resize(self, factor: float) -> None:
        """
        Re-scales the annotations by a factor f. If the layer is None, all
        layers are rescaled, otherwise only the specified layer is rescaled.
        """
        self._mpp /= factor
        self._image_shape['width'] *= factor
        self._image_shape['height'] *= factor

        for obj in self._annots:  # for all objects in layer
            obj.resize(factor)
        return

    def set_resolution(self, mpp: float) -> None:
        """Scales the annotation to the desired mpp.

        :param mpp: (float) target mpp
        """
        if mpp != self._mpp:
            f = self._mpp / mpp
            self.resize(f)
            self._mpp = mpp

        return

    def asdict(self) -> dict:
        d = {'name': self._name,
             'image_shape': self._image_shape,
             'mpp': self._mpp,
             'annotations': [a.asdict() for a in self._annots]
             }

        return d

    def fromdict(self, d: dict) -> None:
        self._name = d['name']
        self._image_shape = d['image_shape']
        self._mpp = d['mpp']

        self._annots.clear()
        for a in d['annotations']:
            obj = createEmptyAnnotationObject(a['annotation_type'])
            obj.fromdict(a)
            self.add_annotation_object(obj)

        return

    def asGeoJSON(self) -> dict:
        """Creates a dictionary compliant with GeoJSON specifications."""

        # GeoJSON does not allow for FeatureCollection properties, therefore
        # we save mpp and image extent as properties of individual
        # features/annotation objects.

        all_annots = []
        for a in self._annots:
            b = a.asGeoJSON()
            b["properties"]["mpp"] = self._mpp
            b["properties"]["image_shape"] = self._image_shape
            all_annots.append(b)

        return gj.FeatureCollection(all_annots)

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize an annotation from a dictionary compatible with GeoJSON specifications."""
        if d["type"].lower() != "featurecollection":
            raise RuntimeError("Need a FeatureCollection as annotation! Got: " + d["type"])

        self._annots.clear()
        mg, im_shape = None, None
        for a in d["features"]:
            obj = createEmptyAnnotationObject(a["geometry"]["type"])
            obj.fromGeoJSON(a)
            self.add_annotation_object(obj)
            if mg is None and "properties" in a:
                mg = a["properties"]["mpp"]
            if im_shape is None and "properties" in a:
                im_shape = a["properties"]["image_shape"]
        self._mpp = mg
        self._image_shape = im_shape

        return

    def save(self, filename: Union[str|Path]) -> None:
        with open(filename, 'wb') as f:
            j_pack({
                'name': self._name,
                'image_shape': self._image_shape,
                'mpp': self._mpp,
                'annotations': self._annots
            }, f)

    def load(self, filename: Union[str|Path]) -> None:
        self._annots.clear()
        with open(filename, 'rb') as f:
            d = j_unpack(f)
            self._name = d['name']
            self._image_shape = d['image_shape']
            self._mpp = d['mpp']
            self._annots = d['annotations']
##-




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
            points_lines.append([points_idx, a.y(), a.x()])
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