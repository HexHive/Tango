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
//   Setup a game, startup stuff.
//


#ifndef __P_SETUP__
#define __P_SETUP__

#include "doomdef.h"
#include "d_player.h"
#include "m_fixed.h"
#include "tables.h"

#define TANGOFUZZ_FEEDBACK_SHM  "/tangofuzz_feedback"
#define TANGOFUZZ_FEEDBACK_SIZE sizeof(tf_feedback_t)

typedef struct {
  float x;
  float y;
  float z;
  float angle;

  playerstate_t playerstate;
  int health; // to be copied from player->mo->health
  int armorpoints;

  boolean cards[NUMCARDS];
  int weaponowned[NUMWEAPONS];
  int ammo[NUMAMMO];

  // mobj_t* attacker;
  boolean attacker_valid;
  float attacker_x;
  float attacker_y;
  float attacker_z;
  boolean didsecret;
  boolean canactivate;
} __attribute__((packed)) tf_feedback_t;

extern tf_feedback_t *tf_feedback;

// NOT called by W_Ticker. Fixme.
void
P_SetupLevel
( int		episode,
  int		map,
  int		playermask,
  skill_t	skill);

// Called by startup code.
void P_Init (void);

#endif
