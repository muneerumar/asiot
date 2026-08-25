"""Message-bus extension boundary.

The evaluated simulator uses equation-level communication delay and resource
costs, not packet- or broker-level MQTT emulation. The empty module makes that
scope boundary explicit without implying an implemented transport.
"""
