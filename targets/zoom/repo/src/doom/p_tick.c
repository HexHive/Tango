//
// Copyright(C) 1993-1996 Id Software, Inc.
// Copyright(C) 2005-2014 Simon Howard
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// DESCRIPTION:
//	Archiving: SaveGame I/O.
//	Thinker, Ticker.
//


#include "z_zone.h"
#include "p_local.h"
#include "p_setup.h"

#include "doomstat.h"

#include <string.h>
// #include <math.h>

int	leveltime;

//
// THINKERS
// All thinkers should be allocated by Z_Malloc
// so they can be operated on uniformly.
// The actual structures will vary in size,
// but the first element must be thinker_t.
//



// Both the head and tail of the thinker list.
thinker_t	thinkercap;


//
// P_InitThinkers
//
void P_InitThinkers (void)
{
    thinkercap.prev = thinkercap.next  = &thinkercap;
}




//
// P_AddThinker
// Adds a new thinker at the end of the list.
//
void P_AddThinker (thinker_t* thinker)
{
    thinkercap.prev->next = thinker;
    thinker->next = &thinkercap;
    thinker->prev = thinkercap.prev;
    thinkercap.prev = thinker;
}



//
// P_RemoveThinker
// Deallocation is lazy -- it will not actually be freed
// until its thinking turn comes up.
//
void P_RemoveThinker (thinker_t* thinker)
{
  // FIXME: NOP.
  thinker->function.acv = (actionf_v)(-1);
}



//
// P_AllocateThinker
// Allocates memory and adds a new thinker at the end of the list.
//
void P_AllocateThinker (thinker_t*	thinker)
{
}



//
// P_RunThinkers
//
void P_RunThinkers (void)
{
    thinker_t *currentthinker, *nextthinker;

    currentthinker = thinkercap.next;
    while (currentthinker != &thinkercap)
    {
	if ( currentthinker->function.acv == (actionf_v)(-1) )
	{
	    // time to remove it
            nextthinker = currentthinker->next;
	    currentthinker->next->prev = currentthinker->prev;
	    currentthinker->prev->next = currentthinker->next;
	    Z_Free(currentthinker);
	}
	else
	{
	    if (currentthinker->function.acp1)
		currentthinker->function.acp1 (currentthinker);
            nextthinker = currentthinker->next;
	}
	currentthinker = nextthinker;
    }
}

mobj_t*     usething;
boolean canuse;

boolean PTR_CanUseTraverse (intercept_t* in)
{
    int     side;
    canuse = false;

    if (!in->d.line->special)
    {
        P_LineOpening (in->d.line);
        if (openrange <= 0)
        {
            // can't use through a wall
            return false;
        }
        // not a special line, but keep checking
        return true ;
    }

    side = 0;
    if (P_PointOnLineSide (usething->x, usething->y, in->d.line) == 1)
        side = 1;

    canuse = true;

    // can't use for than one special line in a row
    return false;
}

void P_CanUseLines (player_t*  player) 
{
    int     angle;
    fixed_t x1;
    fixed_t y1;
    fixed_t x2;
    fixed_t y2;
    
    usething = player->mo;
        
    angle = player->mo->angle >> ANGLETOFINESHIFT;

    x1 = player->mo->x;
    y1 = player->mo->y;
    x2 = x1 + (USERANGE>>FRACBITS)*finecosine[angle];
    y2 = y1 + (USERANGE>>FRACBITS)*finesine[angle];
    
    P_PathTraverse ( x1, y1, x2, y2, PT_ADDLINES, PTR_CanUseTraverse );
}

//
// P_Ticker
//

void P_Ticker (void)
{
    int		i;
    
    // run the tic
    if (paused)
	return;
		
    // pause if in menu and at least one tic has been run
    if ( !netgame
	 && menuactive
	 && !demoplayback
	 && players[consoleplayer].viewz != 1)
    {
	return;
    }
    
		
    for (i=0 ; i<MAXPLAYERS ; i++)
	if (playeringame[i])
	    P_PlayerThink (&players[i]);
			
    P_RunThinkers ();
    P_UpdateSpecials ();
    P_RespawnSpecials ();

    // for par times
    leveltime++;

    // TANGOFUZZ advertise state of players[0] to fuzzer
    if (playeringame[0]) {
        tf_feedback->x = FRACTOFLOAT(players[0].mo->x);
        tf_feedback->y = FRACTOFLOAT(players[0].mo->y);
        tf_feedback->z = FRACTOFLOAT(players[0].mo->z);
        tf_feedback->angle = ANGTODEGREE(players[0].mo->angle);

        tf_feedback->playerstate = players[0].playerstate;
        tf_feedback->health = players[0].mo->health;
        tf_feedback->armorpoints = players[0].armorpoints;

        tf_feedback->attacker_valid = (players[0].attacker != NULL && \
                                        players[0].attacker != players[0].mo && \
                                        players[0].attacker->health > 0);
        if (tf_feedback->attacker_valid) {
            tf_feedback->attacker_x = FRACTOFLOAT(players[0].attacker->x);
            tf_feedback->attacker_y = FRACTOFLOAT(players[0].attacker->y);
            tf_feedback->attacker_z = FRACTOFLOAT(players[0].attacker->z);
            // tf_feedback->attacker_angle = atan2(
            //                     FRACTOFLOAT(players[0].attacker->y - players[0].mo->y),
            //                     FRACTOFLOAT(players[0].attacker->x - players[0].mo->x)
            //                 ) * 180 / 3.14159265;
        }

        memcpy(tf_feedback->cards, players[0].cards, sizeof(players[0].cards));
        memcpy(tf_feedback->weaponowned, players[0].weaponowned, sizeof(players[0].weaponowned));
        memcpy(tf_feedback->ammo, players[0].ammo, sizeof(players[0].ammo));

        tf_feedback->didsecret = players[0].didsecret;
        P_CanUseLines(&players[0]);
        tf_feedback->canactivate = canuse;
    }
}