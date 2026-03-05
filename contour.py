#!/usr/bin/env python3
# Francisco Gutierrez
# For SYN100:Group 6 (with James D.)
# Date 02/2024
# Description: Takes in information from an overlay and creates a .kml with a polygon overlaying a colored area specificed by RGB values
# TODO: Make sure the map completely covers seperately correctly
# TODO: Be able to write into separate files and possible kmz per Areas
# TODO: Add a better way to interface with this
#       TODO: Choose hsv lower & upper or rgb & theta
# TODO: At this point make the kml an object bro
# TODO: The rotation is absolutely horrendous. Takes too much compute. Happens all in tranform_coordinates() but for now satisfied with the speed
# TODO: Determine the largest area using some python library for the shoelace formula, (Gauss's area formula)
#           For now it just checks the largest amount of points when sorting.
# FIXME: Does not account for the poles or where west meets east


import cv2 # handles the coloring and image processing
try:
    import lxml  # noqa: F401 — required by BeautifulSoup's 'xml' parser
except ImportError:
    print("Missing dependency: install lxml with  pip install lxml", file=sys.stderr)
    raise
from bs4 import BeautifulSoup # Note to self, way better than ElementTree
# For the color select
import numpy as np
import math
# Just for the preview of the image
import matplotlib.pyplot as plt
# For accessing arguments
import sys
import os # for some minor file path ( I really want to avoid this one )

AREAS=1; # set of disjoint areas for a specific color
RGBCOLOR = np.uint8([[[  0, 0, 200]]])  # bgr color to select (rgb order is backwards)
OPACITY=200 # Default Opacity of our polygon (0-255), 200 is approx 78%
THETA=15  # hsv, hue +/- angle THETA (Larger number accepts larger color range)
DELTA=0.02 # Polygon simplification factor (fraction of contour perimeter). Lower = more detail.
SAVE_IMAGE=False # save our mask as a .png?

PREVIEW_MASK=False # Preview our overlay on top of old image to see if it looks right
if (PREVIEW_MASK):
    PREVIEW_MASK_ON_IMAGE=True
    QUIT_UPON_PREVIEW=True




GOOGLE_KML_POLYGON_TEMPLATE = ( # Creating one polygon in google earth pro gives this file
'<?xml version="1.0" encoding="UTF-8"?>'
'<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">'
'<Document>'
'	<name>polygon-test.kml</name>' # insert name change here
'	<StyleMap id="m_ylw-pushpin">'
'		<Pair>'
'			<key>normal</key>'
'			<styleUrl>#s_ylw-pushpin</styleUrl>'
'		</Pair>'
'		<Pair>'
'			<key>highlight</key>'
'			<styleUrl>#sn_ylw-pushpin_hl</styleUrl>'
'		</Pair>'
'	</StyleMap>'
'	<Style id="sn_ylw-pushpin">'
'		<IconStyle>'
'			<scale>1.1</scale>'
'			<Icon>'
'				<href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href>'
'			</Icon>'
'			<hotSpot x="20" y="2" xunits="pixels" yunits="pixels"/>'
'		</IconStyle>'
'       <LineStyle>'
'       	<color></color>' # color of polygon perimeter lines goes here
'       </LineStyle>'
'       <PolyStyle>'
'       	<color></color>' # color of area of polygon goes here
'       </PolyStyle>'
'	</Style>'
'	<Style id="s_ylw-pushpin_hl">'
'		<IconStyle>'
'			<scale>1.3</scale>'
'			<Icon>'
'				<href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href>'
'			</Icon>'
'			<hotSpot x="20" y="2" xunits="pixels" yunits="pixels"/>'
'		</IconStyle>'
'	</Style>'
'	<Placemark>'
'		<name>polygon-test</name>' # insert name of polygon here
'		<styleUrl>#sn_ylw-pushpin</styleUrl>' # here it adopts the style we want (color)
'		<Polygon>'
'			<tessellate>1</tessellate>'
'			<outerBoundaryIs>'
'				<LinearRing>'
'					<coordinates>' # put coordinate in this tag and end with the same starting coordinate
'					</coordinates>'
'				</LinearRing>'
'			</outerBoundaryIs>'
'		</Polygon>'
'	</Placemark>'
'</Document>'
'</kml>'
)
SUFFICIENT_KML_TEMPLATE = ( # I just made my own too this because I'm salty
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<kml xmlns="http://www.opengis.net/kml/2.2">'
    '<Document>'
    '  <name>fmars-red.kml</name>'
    '  <open>0</open>'
    '  <Placemark>'
    '    <name>hollow box</name>'
    '    <Polygon>'
    '      <extrude>1</extrude>'
    '      <altitudeMode>relativeToGround</altitudeMode>'
    '      <outerBoundaryIs>'
    '        <LinearRing>'
    '          <coordinates>'
    '          </coordinates>'
    '        </LinearRing>'
    '      </outerBoundaryIs>'
    '    </Polygon>'
    '  </Placemark>'
    '</Document>'
    '</kml>'
)



def extractDataFromKML(kmlFile):
    '''
    Gets the data from the input KML file. Assumes one overlay is saved with one image. Returns nothing but edits global variables, promarily the imageName and the variables pertaining to the placement of the overlay (bounds and rotation).
    @Parameter coord, a pair (as a list) of numbers denoting lat and long
    '''
    try:
        # Load the KML file
        with open(kmlFile, 'r', encoding='utf-8') as overlay:
            kml_data = overlay.read()

        # Parse the KML data using BeautifulSoup
        soup = BeautifulSoup(kml_data, 'xml')

        name = soup.find('name')
        # Since rotation is completely optional
        if (name == None):
            print("Extracting <name> from kml yielded {}!".format(name), file=sys.stderr)
            quit()
        name = name.text

        # Find the <north>, <east>, <south>, <west>, and <rotation> elements
        # By converting to float we likely lose some precision
        n = float(soup.find('north').text)
        e = float(soup.find('east').text)
        s = float(soup.find('south').text)
        w = float(soup.find('west').text)
        image_location = soup.find('href').text
        r = soup.find('rotation')
        # Since rotation is completely optional
        if (r != None):
            r = float(r.text)
        else:
            r = 0;

        return (name, image_location, n, s, e, w, r)

    except FileNotFoundError:
        print("File not found: ", kmlFile)
        quit()


# Bounds
def transform_coordinates(coords, hsv, center, bounds, rotation):
    '''
    Transforms numpy array of coordinates using image data (hsv) and our overlay center+bounds+rotation our polygon right
    @Parameter coords, a numpy array of coordinates of an area
    @Parameter hsv, the original image mask which is just used for helping find coordinates
    @Parameter center, a pair (tuple) denoting (x,y) center of the image
    @Parameter bounds, pair (tuple) denoting width and height of the overlay on lat-long map scale
    @Parameter rotation, a float giving the angle of rotation in degrees of the overlay
    '''
    # Convert image to HSV color space
    max_image_h, max_image_w, _ = hsv.shape
    # change all of our points from top right to centered
    # Also notice that the y axis is still backwards here, but at least it is centered
    coords = np.array([-max_image_w/2 , -max_image_h/2] + coords, dtype=np.longdouble)

    angle = np.deg2rad(rotation-2, dtype=np.longdouble)  # Cast angle to long double
    #!FIXME THE ROTATION IS A LITTLE OFF
    ROTATIONAL_MATRIX = np.array([[np.cos(angle, dtype=np.longdouble), -np.sin(angle, dtype=np.longdouble)],
                  [np.sin(angle, dtype=np.longdouble),  np.cos(angle, dtype=np.longdouble)]], dtype=np.longdouble)
    coords =  coords.dot(ROTATIONAL_MATRIX) # Apply rotational matrix
    np_scale_factor = np.array(
            [# extra precision (not actually useful but alright)
                np.divide(np.longdouble(bounds[0]),np.longdouble(max_image_w)),
                np.divide(np.longdouble(-bounds[1]),np.longdouble(max_image_h))
            ],
            dtype=np.longdouble);

    final_coords = coords * np_scale_factor # Scale Image
    final_coords = np.array([center[0], center[1]] + final_coords , dtype=np.longdouble)# Move 
    
    return final_coords
    


def writeKML(filename, coords, rgb):
    '''
    Writes the KML polygon file. As of now, it mostly just calls writes the coordinates and dips
    @Parameter filename, a string with the output name of our file .kml
    @Parameter coords, list of coordinates for our file
    @Parameter rgb, array of rgb values
    '''
    # Parse XML content
    soup = BeautifulSoup(GOOGLE_KML_POLYGON_TEMPLATE, 'xml')

    # Now edit the name
    polygon_kml_name = soup.find('Document').find('name')
    polygon_kml_name.string = "{}".format(filename.removesuffix('.kml'))
    # Edit the Placemark name
    polygon_name = soup.find('Placemark').find('name')
    polygon_name.string = "{}-polygon".format(filename.removesuffix('.kml'))

    # Add the coordinates tag
    coordinates_tag = soup.find("coordinates")
    altitude = 0;
    coordinates_tag.string = '\n\t'.join(','.join(map(str, pair + [altitude])) for pair in coords) # this looks cursed but suggest by gpt


    # Add a color and opacity
    line_color = soup.find("Style", id="sn_ylw-pushpin").find("LineStyle").find("color");
    area_color = soup.find("Style", id="sn_ylw-pushpin").find("PolyStyle").find("color");
    line_color.string = "{:02x}{:02x}{:02x}{:02x}".format(OPACITY,rgb[0][0][0],rgb[0][0][1],rgb[0][0][2]);
    area_color.string = "{:02x}{:02x}{:02x}{:02x}".format(OPACITY,rgb[0][0][0],rgb[0][0][1],rgb[0][0][2]); 
    
    # Convert the modified XML tree to string
    modified_xml = soup.prettify().replace('kml:', '')
    #print(modified_xml) # debug

    # Write the resulting kml
    print("Writing polygon to {}".format(filename))
    with open(filename, "w") as file:
        file.write(modified_xml)



def findMaskBounds(rgb_color, theta=10):
    '''
    Get's the color from rgb to hsv. 
    @Parameter rgb_color: Takes in a np array denoting a rgb value
    @OptionalParameter delta: The leeway given for the final hsv value, +/- delta
    @Returns a pair of hsv values, a lower and a upper bound value for hsvValue for later recognition.
    '''
    hsvColor = cv2.cvtColor(rgb_color, cv2.COLOR_BGR2HSV)
    # Saturation/value floors at 50 to accept darker or less-saturated shades of the color.
    # Previously these were 100, which was too strict and missed valid pixels.
    lowerbound = np.array([hsvColor[0][0][0] - theta, 50, 50])  # look at hsv cone/cyl
    upperbound = np.array([hsvColor[0][0][0] + theta, 255, 255])
    return lowerbound, upperbound
    
    
    

def main():
    # This could go easily go wrong but assumes user input in correct format
    input_kml = sys.argv[1]
    base_path = os.path.dirname(input_kml)
    # Find the overlay *.kml file and extract information:
    name, image_name, n, s, e, w, r = extractDataFromKML(input_kml)
    center = ((w+e)/2, (n+s)/2)
    bounds = (e-w, n-s)

    
    # We assume the basepath of the input_kml and image are the same
    # FIXME: Add support for windows here by changing how filepath works
    if (base_path == ''):
        base_path = '.'
    image_path="{}/{}".format(base_path,image_name)
    output_kml="poly-{}.kml".format(name)

    # Load the image
    image = cv2.imread(image_path)
    
    # Convert image to HSV color space
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    except:
        print("\nException Caught. Possibly because from the kml file input kml {}"
        "\n     It tried to take basepath {}"
        "\n     And concluded could not find image at: {}"
        "\n     Believe the path was {}, but it was not".format(input_kml, base_path, image_name, image_path))
        quit()
    
    
    # Good way to set a color range https://stackoverflow.com/questions/36817133/identifying-the-range-of-a-color-in-hsv-using-opencv
    
    
    # Assumes a bright color for RGBCOLOR
    lowerbound, upperbound = findMaskBounds(RGBCOLOR, THETA);
    
    
    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lowerbound, upperbound)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Simplify every contour in-place. DELTA controls how closely the polygon
    # follows the original shape — smaller = more faithful, larger = smoother.
    contours = [
        cv2.approxPolyDP(c, DELTA * cv2.arcLength(c, True), True)
        for c in contours
    ]
    
    # Create a blank image to draw contours
    contour_image = np.zeros_like(image, np.uint8)
    
    # You can use this to look at the contours only if you'd like, though I forget already what looks like
    #cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)  # You can adjust the color and thickness here
    
    
    if not contours:
        print(
            "No contours found. The color mask matched zero pixels.\n"
            "  - Check that RGBCOLOR matches a color actually present in the image.\n"
            "  - Try increasing THETA (currently {}) to widen the hue range.".format(THETA),
            file=sys.stderr,
        )
        quit()

    # Rank contours by actual pixel area (cv2.contourArea) rather than point count.
    # Point count was a poor proxy — a jagged small region can have more points than a
    # large smooth one. Sorting by area gives the true AREAS largest regions.
    contour_areas = [(cv2.contourArea(c), idx) for idx, c in enumerate(contours)]
    contour_areas.sort(key=lambda x: x[0], reverse=True)
    max_contour_indices = [idx for _, idx in contour_areas[:AREAS]]
    
    #TODO: Assert Areas is at least 1 so this is not empty
    # Display the contour image
    for i in range(AREAS):
        contour_image = cv2.fillPoly(contour_image, [contours[max_contour_indices[-i]].reshape((-1, 1, 2))], (255, 255, 255))
    
    if (PREVIEW_MASK):
        if (PREVIEW_MASK_ON_IMAGE):
            preview = image.copy();
            # show original image with semi-transparent mask
            alpha=0.35
            mask = contour_image.astype(bool)

            preview[mask] = cv2.addWeighted(image, alpha, contour_image, 1 - alpha, 0)[mask]
        else:
            preview = contour_image
        
        print("Previewing")
        plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        if (QUIT_UPON_PREVIEW):
            quit()

    if (SAVE_IMAGE):
        # takes the output kml file and changes extension to .png
        print("Saving image to {}".format(output_kml.rsplit('.', 1)[0] + ".png"))
        cv2.imwrite(output_kml.rsplit('.', 1)[0] + ".png",contour_image)
    
    # Combine Areas
    close_points = []
    final_coords = None
    for i in max_contour_indices:
        coord_numpy = np.array(contours[i]).reshape(-1,2)
        # For each contour/area we want the points to end where they started
        close_points.append(coord_numpy[0])
        coord_numpy = np.append(coord_numpy, [coord_numpy[0]], axis=0)

        if ( np.any(final_coords) ):
            final_coords = np.append(final_coords, coord_numpy, axis=0)
        else:
            final_coords = coord_numpy

    # Make sure each Area begins where it ends (repeats a point thrice now)
    for point in close_points[::-1]:
        final_coords = np.append(final_coords, [point], axis=0)

    final_coords = np.append(final_coords, [final_coords[0]], axis=0)

    # change our final_coords to lat-long)
    final_coords = transform_coordinates(final_coords, hsv, center, bounds, r)

    # Create the KML coordinates
    writeKML(output_kml, final_coords, RGBCOLOR);
    quit() # we are done
            
    

if __name__ == "__main__":
    main()
