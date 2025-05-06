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
## The annotation objects belong to at least one group: "no_group".
## Other groups may be added, and objects may belong to several groups.
##
## All annotations share the same underlying mesh (= a raster of pixels with
## predefined extent and fixed resolution (microns-per-pixels)).
##
## The coordinates of the various objects (or parts of them) are specified in
## as (X,Y) pairs (horizontal and vertical) and not in (row, column) system.

__all__ = ['AnnotationObject', 'Dot', 'Polygon', 'PointSet', 'Annotation', 'Circle']

import io
from abc import ABC, abstractmethod
from typing import Union, Optional
from pathlib import Path
import shapely
import shapely.geometry as shg
import shapely.affinity as sha
import geojson as gj
import numpy as np
import collections
from ._serialize import pack as j_pack, unpack as j_unpack
import pandas as pd
import orjson as json

##-
class AnnotationObject(ABC):
    """Define the AnnotationObject minimal interface. This class is made
    abstract to force more meaningful names (e.g. Dot, Polygon, etc.) in
    subclasses."""

    def __init__(self, coords: Union[list|tuple],
                 name: Optional[str] = None,
                 group: Optional[list] = None,
                 data: Optional[pd.DataFrame] = None):
        # main geometrical object describing the annotation:

        self._geom = shg.Point()  # some empty geometry
        self._name = name
        self._annotation_type = None
        if isinstance(group, list):
            self._metadata = {'group': group}
        else:
            self._metadata = {"group": ["no_group"]}
        if isinstance(data, pd.DataFrame):
            self._data = data
        else:
            self._data = None # None or a data frame with measurements associated with the annotation

        return

    @abstractmethod
    def duplicate(self):
        pass

    @property
    def data(self):
        return self._data

    def set_data(self, data: Union[None, pd.DataFrame]):
        self._data = data

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
        scaling with the origin set to (0,0) and the same factor for both x and y
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

    @property
    def x(self):
        """Return the x coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return shapely.get_coordinates(self.geom)[:,0]

    @property
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
        d = {
            "annotation_type": self._annotation_type,
            "name": self._name,
            "metadata": self._metadata,
            "data": self._data.to_dict('tight') if self._data is not None else [],
            "geom": shapely.to_wkt(self.geom)   # text representation of the geometry
        }

        return d

    def fromdict(self, d: dict) -> None:
        """Initialize the object from a dictionary."""

        self._annotation_type = d["annotation_type"]
        self._name = d["name"]
        self._metadata = d["metadata"]
        self._data = pd.DataFrame(d["data"]) if isinstance(d["data"], dict) else None
        self._geom = shapely.from_wkt(d["geom"])

        return

    def asGeoJSON(self) -> dict:
        """Return a dictionary compatible with GeoJSON specifications."""
        return gj.Feature(geometry=shg.mapping(self.geom),
                          properties=dict(object_type="annotation",
                                          annotation_type=self._annotation_type,
                                          name=self._name,
                                          metadata=self._metadata,
                                          data=self._data.to_dict(as_series=False) if self._data is not None else [])
                          )

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        """This is a basic function - further tests should be implemented for particular object
        types."""

        self._geom = shg.shape(d["geometry"])
        try:
            self._name = d["properties"]["name"]
            self._metadata = d["properties"]["metadata"]
            if isinstance(d["properties"]["data"], dict):
                self._data = pd.DataFrame(d["properties"]["data"])
            else:
                self._data = None
        except KeyError:
            pass
##-


##-
class Dot(AnnotationObject):
    """Dot: a single position in the image."""

    def __init__(self, coords=(0.0, 0.0),
                 name: Optional[str] = None,
                 group: Optional[list] = None,
                 data: Optional[pd.DataFrame] = None):
        """Initialize a DOT annotation, i.e. a single point in plane.

        Args:
            coords (list or vector or tuple): the (x,y) coordinates of the point
            name (str): the name of the annotation
        """
        super().__init__(coords, name, group, data)

        self._annotation_type = "DOT"
        self._name = "DOT" if name is None else name

        if not isinstance(coords, collections.abc.Iterable):
            raise RuntimeError('coords parameter cannot be interpreted as a 2D vector')

        self._geom = shg.Point(coords)

        return

    def duplicate(self):
        return Dot(shapely.get_coordinates(self.geom), name=self.name, group=self.group, data=self.data)

    def size(self) -> int:
        """Return the number of points defining the object."""
        return 1

    # def fromdict(self, d: dict) -> None:
    #     super().fromdict(d)
    #     self._geom = shg.Point((d["x"], d["y"]))
    #
    #     return

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

    def __init__(self, coords: Union[list|tuple],
                 name: Optional[str] = None,
                 group: Optional[list] = None,
                 data: Optional[pd.DataFrame] = None):
        """Initialize a POINTSET annotation, i.e. a collection
         of points in plane.

        Args:
            coords (list or tuple): coordinates of the points as in [(x0,y0), (x1,y1), ...]
            name (str): the name of the annotation
        """

        super().__init__(coords, name, group, data)
        self._annotation_type = "POINTSET"
        self._name = "POINTS" if name is None else name

        # check whether coords is iterable and build the coords from it:
        if not isinstance(coords, collections.abc.Iterable):
            raise RuntimeError('coords parameter cannot be interpreted as a 2D array')

        self._geom = shg.MultiPoint(coords)

        return

    def duplicate(self):
        return PointSet(shapely.get_coordinates(self.geom), name=self.name, group=self.group, data=self.data)

    def size(self) -> int:
        """Return the number of points defining the object."""
        return self.xy().shape[0]

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

    def __init__(self, coords: Union[list|tuple],
                 name: Optional[str] = None,
                 group: Optional[list] = None,
                 data: Optional[pd.DataFrame] = None):
        """Initialize a POLYLINE object.

        Args:
            coords (list or tuple): coordinates of the points [(x0,y0), (x1,y1), ...]
                defining the segments (x0,y0)->(x1,y1); (x1,y1)->(x2,y2),...
            name (str): the name of the annotation
        """
        super().__init__(coords, name, group, data)
        self._annotation_type = "POLYLINE"

        # check whether x is iterable and build the coords from it:
        if not isinstance(coords, collections.abc.Iterable):
            raise RuntimeError('coords parameter cannot be interpreted as a 2D array')

        self._geom = shg.LineString(coords)

        return

    def duplicate(self):
        return PolyLine(shapely.get_coordinates(self.geom), name=self.name, group=self.group, data=self.data)

    def size(self) -> int:
        """Return the number of points defining the object."""
        return self.xy().shape[0]

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

    def __init__(self, coords: Union[list|tuple],
                 name: Optional[str] = None,
                 group: Optional[list] = None,
                 data: Optional[pd.DataFrame] = None):
        """Initialize a POINTSET annotation, i.e. a collection
         of points in plane.

        Args:
            coords (list or tuple): coordinates of the points as in [(x0,y0), (x1,y1), ...]
            name (str): the name of the annotation
        """

        super().__init__(coords, name, group, data)
        self._annotation_type = "POLYGON"
        self._name = name if not None else "POLYGON"

        # check whether x is iterable and build the coords from it:
        if not isinstance(coords, collections.abc.Iterable):
            raise RuntimeError('coords parameter cannot be interpreted as a 2D array')

        self._geom = shg.Polygon(coords)

        return

    def duplicate(self):
        return Polygon(shapely.get_coordinates(self.geom), name=self.name, group=self.group, data=self.data)

    def size(self) -> int:
        """Return the number of points defining the object."""
        return self.xy().shape[0]

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
    def __init__(self, center: Union[list|tuple], radius: float,
                 name: Optional[str] = None,
                 group: Optional[list] = None,
                 data: Optional[pd.DataFrame] = None,
                 n_points: int = 8):
        alpha = np.array([k*2*np.pi/n_points for k in range(n_points)])
        coords = np.vstack((
            radius * np.sin(alpha) + center[0], radius * np.cos(alpha) + center[1]
        )).transpose()

        super().__init__(coords.tolist(), name, group, data)
        self._annotation_type = "CIRCLE"

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        d = super().asdict()
        d["radius"] = self.radius
        d["center"] = self.center

        return d

    @property
    def center(self):
        return self.geom.centroid.coords

    @property
    def radius(self):
        return shapely.minimum_bounding_radius(self.geom)


    def fromdict(self, d: dict) -> None:
        """Initialize the object from a dictionary."""
        super().fromdict(d)
        self._annotation_type = "CIRCLE"

        return

    def asGeoJSON(self) -> dict:
        """Return a dictionary compatible with GeoJSON specifications."""
        return gj.Feature(geometry=shg.mapping(self.geom),
                          properties=dict(object_type="annotation",
                                          annotation_type=self._annotation_type,
                                          name=self._name,
                                          metadata=self._metadata,
                                          data=self._data.to_dict(as_series=False) if self._data is not None else [],
                                          radius=self.radius,
                                          center=self.center)
                          )

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
    coordinate system (mesh) and grouped in layers.
    """

    def __init__(self, name: str = "", image_shape=None, mpp: float = 1.0) -> None:
        """Initialize an Annotation for a slide.

        :param name: (str) name of the annotation
        :param image_shape: (dict) shape of the image corresponding to the annotation
            {'width':..., 'height':...}
        :param mpp: (float) slide resolution (microns-per-pixel) for the image
        """
        self._name = name
        if image_shape is None:
            image_shape = dict(width=0, height=0)
        else:
            if 'width' not in image_shape or 'height' not in image_shape:
                raise RuntimeError('Invalid shape specification (<width> or <height> key missing)')
        self._image_shape = image_shape
        self._annots = {'base': []}
        self._mpp = mpp

        self._image_shape = image_shape

        return

    def add_annotation_object(self, a: AnnotationObject, layer: Optional[str] = 'base') -> None:
        if layer in self._annots:
            self._annots[layer].append(a)
        else:
            self._annots[layer] = [a]

    def add_annotations(self, a: list, layer: Optional[str] = 'base') -> None:
        if layer in self._annots:
            self._annots[layer].extend(a)
        else:
            self._annots[layer] = a

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
        Re-scales all annotations (in all layers) by a factor f.
        """
        self._mpp /= factor
        self._image_shape['width'] *= factor
        self._image_shape['height'] *= factor

        for ly in self._annots:  # for all layers
            for obj in self._annots[ly]:  # for all objects in the current layer
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
             'annotations': {l: [a.asdict() for a in self._annots[l]] for l in self._annots}
             }

        return d

    def fromdict(self, d: dict) -> None:
        self._name = d['name']
        self._image_shape = d['image_shape']
        self._mpp = d['mpp']

        self._annots.clear()
        for ly in d['annotations']:
            for a in d['annotations'][ly]:
                obj = createEmptyAnnotationObject(a['annotation_type'])
                obj.fromdict(a)
                self.add_annotation_object(obj, layer=ly)

        return

    # def asGeoJSON(self) -> dict:
    #     """Creates a dictionary compliant with GeoJSON specifications."""
    #
    #     # GeoJSON does not allow for FeatureCollection properties, therefore
    #     # we save mpp and image extent as properties of individual
    #     # features/annotation objects.
    #
    #     all_annots = []
    #     for a in self._annots:
    #         b = a.asGeoJSON()
    #         b["properties"]["mpp"] = self._mpp
    #         b["properties"]["image_shape"] = self._image_shape
    #         all_annots.append(b)
    #
    #     return gj.FeatureCollection(all_annots)
    #
    # def fromGeoJSON(self, d: dict) -> None:
    #     """Initialize an annotation from a dictionary compatible with GeoJSON specifications."""
    #     if d["type"].lower() != "featurecollection":
    #         raise RuntimeError("Need a FeatureCollection as annotation! Got: " + d["type"])
    #
    #     self._annots.clear()
    #     mg, im_shape = None, None
    #     for a in d["features"]:
    #         obj = createEmptyAnnotationObject(a["geometry"]["type"])
    #         obj.fromGeoJSON(a)
    #         self.add_annotation_object(obj)
    #         if mg is None and "properties" in a:
    #             mg = a["properties"]["mpp"]
    #         if im_shape is None and "properties" in a:
    #             im_shape = a["properties"]["image_shape"]
    #     self._mpp = mg
    #     self._image_shape = im_shape
    #
    #     return

    def save_binary(self, filename: Union[str|Path]) -> None:
        """Save the annotation as nested dictionaries into a binary file."""
        with open(filename, 'wb') as f:
            j_pack(self.asdict(), f)

    def load_binary(self, filename: Union[str|Path]) -> None:
        """Load the annotation from a binary file."""
        self._annots.clear()
        with open(filename, 'rb') as f:
            d = j_unpack(f)
            self.fromdict(d)

    def save(self, file_obj:io.IOBase) -> None:
        """Save the annotation into a portable and efficient format."""
        file_obj.write(
            json.dumps(
                self.asdict(),
                option=json.OPT_NON_STR_KEYS | json.OPT_INDENT_2 | json.OPT_SERIALIZE_NUMPY
            )
        )

    def load(self, file_obj: io.IOBase) -> None:
        """Load the annotation from external file."""
        self.fromdict(
            json.loads(file_obj.read())
        )
        
##-
