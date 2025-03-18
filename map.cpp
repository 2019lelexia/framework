#include "map.h"

Map::Map()
{
    frames.clear();
}

Map::~Map()
{}

void Map::addKeyFrameToMap(shared_ptr<Frame> _frame)
{
    if(frames.find(_frame) == frames.end())
    {
        frames.insert(_frame);
    }
}