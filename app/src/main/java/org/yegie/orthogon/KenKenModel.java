/*
 * KenKenModel.java: Core game state model for KenKen puzzles
 *
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (C) 2016 Sergey
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
 */

package org.yegie.orthogon;

import java.util.Stack;

/**
 * Model for a keen game
 */
public class KenKenModel {

    public static final int MAX_SIZE = 9;

    //holds the data about a single grid cell
    public static class GridCell
    {
        public boolean[] guesses;
        public float[] mlProbabilities; // [0] unused, [1..N] probs
        public int finalGuessValue;
        public int expectedValue;
        public Zone zone;

        public GridCell(int eV, Zone zone)
        {
            this.finalGuessValue = -1;
            this.guesses = new boolean[MAX_SIZE];
            this.mlProbabilities = null;
            this.expectedValue = eV;
            this.zone = zone;
        }

        public int getFinalGuessValue() { return finalGuessValue; }
        public boolean[] getGuesses() { return guesses; }
        public Zone getZone() { return zone; }
    }

    //holds data about a single zone (area grouped by bold lines)
    public static class Zone
    {
        public enum Type
        {
            ADD,MINUS,TIMES,DIVIDE,EXPONENT
        }

        public Type zoneType;
        public int expectedValue;
        public int code;

        public Zone(Type type, int eV, int code)
        {
            this.zoneType = type;
            this.expectedValue = eV;
            this.code = code;
        }
        public String toString()
        {
            String a = ""+expectedValue;
            switch (zoneType)
            {
                case ADD:
                    a+=" +";
                    break;
                case MINUS:
                    a+=" -";
                    break;
                case TIMES:
                    a+=" x";
                    break;
                case DIVIDE:
                    a+=" /";
                    break;
                case EXPONENT:
                    a+=" ^";
                    break;
            }

            return a;
        }
    }

    //a more memory friendly cell that is used for the
    //undo stack (which could potentially get quite large)
    private static class CellState{
        /*
         * state pos 0: 0 = not final 1 = final
         * state pos 1-9: 0 = false 1 = true
         */
        short state;
        short x,y;

        CellState(short state, short x, short y){
            this.state = state;
            this.x = x;
            this.y = y;
        }

    }

    //variables that together describe the state of the game
    private final GridCell[][] gameGrid;
    private transient Stack<CellState> undo;  // transient: not serialized, reinit after deserialize
    private final Zone[] gameZones;
    private boolean finalGuess;
    private boolean puzzleWonVal;
    private short activeX;
    private short activeY;
    private int size;
    private boolean wasMlGenerated;

    //constructor that initializes everything to the default values
    public KenKenModel(int size, Zone[] zones, GridCell[][] grid, boolean wasMlGenerated)
    {
        this.size = size;
        this.gameGrid = grid;
        this.undo = new Stack<>();
        this.gameZones = zones;
        this.finalGuess = true;
        this.puzzleWonVal = false;
        this.wasMlGenerated = wasMlGenerated;
        activeX = -1;
        activeY = -1;
    }

    /** Whether the puzzle was generated using ML */
    public boolean wasMlGenerated() { return wasMlGenerated; }

    /** Whether there is an active (selected) cell */
    public boolean hasActiveCords() { return activeX >= 0 && activeY >= 0; }

    /** Get the grid for iteration. Returns the internal array - treat as read-only. */
    public GridCell[][] getGrid() { return gameGrid; }

    /**
     * Ensures transient fields are initialized after Gson deserialization.
     * Call this after loading a saved game from JSON.
     */
    public void ensureInitialized() {
        if (undo == null) {
            undo = new Stack<>();
        }
    }

    //public methods that allow other classes to modify/view the variables
    public void addCurToUndo(short x, short y){
        short state = 1;
        GridCell curCell = gameGrid[x][y];
        if(curCell.finalGuessValue == -1) {
            state = 0;
            for (int i = 1; i <= curCell.guesses.length; ++i) {
                if (curCell.guesses[i - 1]) {
                    state |= (short) (1 << i);
                }
            }
        } else {
            state |= (short) (1<<curCell.finalGuessValue);
        }
        CellState val = new CellState(state,x,y);

        undo.push(val);
    }

    public void undoOneStep(){
        if(!undo.empty()) {
            CellState oldCell = undo.pop();
            short x = oldCell.x;
            short y = oldCell.y;
            GridCell cellToUpdate = gameGrid[x][y];
            boolean finalGuess = ((oldCell.state & 1) != 0);
            clearFinal(x,y);
            clearGuesses(x,y);
            for (int i = 1; i <= cellToUpdate.guesses.length; ++i) {
                if ((oldCell.state & 1 << i) != 0) {
                    if (finalGuess) {
                        cellToUpdate.finalGuessValue = i;
                    } else {
                        cellToUpdate.guesses[i - 1] = true;
                    }
                }
            }
        }
    }

    public short getActiveY(){return activeY;}
    public short getActiveX(){return activeX;}

    public void setActiveX(short activeX) {
        this.activeX = activeX;
    }
    public void setActiveY(short activeY) {
        this.activeY = activeY;
    }

    public void clearGuesses(short x, short y){
        gameGrid[x][y].guesses = new boolean[MAX_SIZE];
    }
    public void clearFinal(short x, short y){
        gameGrid[x][y].finalGuessValue = -1;
    }
    public void setCellFinalGuess(short x, short y, int guess){
        if (guess < 1 || guess > size) return;
        if(gameGrid[x][y].finalGuessValue == guess)
        {
            gameGrid[x][y].finalGuessValue = -1;
        } else {
            gameGrid[x][y].finalGuessValue = guess;
        }
    }

    public void puzzleWon()
    {
        puzzleWonVal = setPuzzleWon();
        if(puzzleWonVal) {
            activeX = -1;
            activeY = -1;
        }
    }
    private boolean setPuzzleWon()
    {
        for(int x = 0; x < size; x++)
            for(int y = 0; y < size; y++)
                if(gameGrid[x][y].finalGuessValue != gameGrid[x][y].expectedValue)
                    return false;
        return true;
    }
    public boolean getPuzzleWon()
    {
        return puzzleWonVal;
    }

    public void addToCellGuesses(short x, short y, int guess){
        if (guess < 1 || guess > size) return;
        gameGrid[x][y].guesses[guess-1] = !gameGrid[x][y].guesses[guess-1];
    }

    public GridCell getCell(short x, short y)
    {
        return gameGrid[x][y];
    }

    public Zone[] getGameZones() {
        return gameZones;
    }

    public int getSize()
    {
        return size;
    }

    public boolean getFinalGuess() {return finalGuess; }
    public void toggleFinalGuess() {finalGuess = !finalGuess;}

}
